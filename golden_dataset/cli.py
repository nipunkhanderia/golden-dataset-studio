"""CLI for Golden Dataset.

Commands:
  golden init <name>        Initialise a new dataset
  golden add                Add a golden entry
  golden ingest <file>      Bulk-import from JSONL / CSV / JSON
  golden status             Show working tree and version history
  golden ls                 List entries in a version
  golden commit             Snapshot working tree as a new version
  golden diff <v1> <v2>     Show what changed between two versions
  golden export             Export a version to JSONL / CSV / JSON
  golden evaluate           Score actual LLM answers vs golden answers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .evaluator import Evaluator
from .exporter import export_csv, export_json, export_jsonl
from .importer import import_csv, import_json, import_jsonl
from .models import GoldenEntry
from .store import DatasetStore

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
def cli(ctx: click.Context) -> None:
    """✨ Golden Dataset — version control for your LLM test data."""
    ctx.ensure_object(dict)
    ctx.obj["store"] = DatasetStore()


@cli.command()
@click.argument("name")
@click.option("--description", "-d", default="", help="Short dataset description")
@click.pass_context
def init(ctx: click.Context, name: str, description: str) -> None:
    """Initialise a new golden dataset in the current directory."""
    store: DatasetStore = ctx.obj["store"]
    try:
        manifest = store.init(name, description)
        console.print(
            f"[bold green]OK[/] Initialised dataset [bold]{manifest.name}[/]"
            f" in [dim].golden_dataset/[/]"
        )
    except FileExistsError as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)


@cli.command()
@click.option("--question", "-q", default=None, help="The question / prompt")
@click.option("--answer", "-a", default=None, help="The golden answer")
@click.option("--context", "-c", multiple=True, help="Context passage (repeatable)")
@click.option("--tag", "-t", multiple=True, help="Tag (repeatable)")
@click.option("--meta", "-m", default=None, help="Metadata as a JSON string")
@click.pass_context
def add(ctx: click.Context, question: str, answer: str, context: tuple, tag: tuple, meta: str) -> None:
    """Add a single golden entry (prompts for missing fields)."""
    store: DatasetStore = ctx.obj["store"]
    question = question or click.prompt("Question")
    answer = answer or click.prompt("Golden answer")

    metadata: dict = {}
    if meta:
        try:
            metadata = json.loads(meta)
        except json.JSONDecodeError:
            console.print("[yellow]Warning:[/] --meta is not valid JSON; ignoring.")

    entry = GoldenEntry(
        question=question,
        answer=answer,
        contexts=list(context),
        tags=list(tag),
        metadata=metadata,
    )
    store.add_entry(entry)
    console.print(f"[green]OK[/] Added entry [bold]{entry.id}[/]")


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["jsonl", "csv", "json"], case_sensitive=False), default=None)
@click.pass_context
def ingest(ctx: click.Context, file: Path, fmt: str | None) -> None:
    """Bulk-import entries from a JSONL, CSV, or JSON file."""
    store: DatasetStore = ctx.obj["store"]
    detected = fmt or file.suffix.lstrip(".").lower()
    importers = {"jsonl": import_jsonl, "csv": import_csv, "json": import_json}
    if detected not in importers:
        console.print(f"[red]Error:[/] Cannot determine format for '{file.name}'. Use --format.")
        sys.exit(1)
    entries = importers[detected](file)
    for e in entries:
        store.add_entry(e)
    console.print(f"[green]OK[/] Imported [bold]{len(entries)}[/] entries from {file.name}")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current working tree and version history."""
    store: DatasetStore = ctx.obj["store"]
    try:
        manifest = store.load_manifest()
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)

    working = store.get_working_entries()
    console.print(f"\n[bold]Dataset:[/]  {manifest.name}")
    if manifest.description:
        console.print(f"[dim]{manifest.description}[/]")
    console.print(
        f"[bold]Version:[/]  {manifest.current_version or '[dim]none[/]'}  "
        f"  [bold]Working tree:[/] {len(working)} entries"
    )

    if manifest.versions:
        table = Table(title="Version History", box=None, show_edge=False)
        table.add_column("Version", style="cyan bold")
        table.add_column("Entries", justify="right")
        table.add_column("Hash", style="dim")
        table.add_column("Message")
        table.add_column("Created", style="dim")
        for v in manifest.versions:
            table.add_row(v.version, str(v.entry_count), v.sha256[:8], v.description or "-", str(v.created_at)[:19])
        console.print()
        console.print(table)
    else:
        console.print("\n[dim]No versions committed yet. Run `golden commit`.[/]")


@cli.command()
@click.option("--message", "-m", default="", help="Commit message")
@click.pass_context
def commit(ctx: click.Context, message: str) -> None:
    """Snapshot the working tree as a new immutable version."""
    store: DatasetStore = ctx.obj["store"]
    try:
        version = store.commit(message)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    console.print(
        f"[green]OK[/] Committed version [bold cyan]{version.version}[/] "
        f"({version.entry_count} entries, hash {version.sha256[:8]})"
    )


@cli.command("ls")
@click.option("--version", "-v", default=None, help="Version to list")
@click.option("--tag", "-t", default=None, help="Filter by tag")
@click.option("--search", "-s", default=None, help="Substring search in question/answer")
@click.pass_context
def ls(ctx: click.Context, version: str | None, tag: str | None, search: str | None) -> None:
    """List entries in a committed version."""
    store: DatasetStore = ctx.obj["store"]
    try:
        entries = store.load_version(version)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)

    if tag:
        entries = [e for e in entries if tag in e.tags]
    if search:
        lo = search.lower()
        entries = [e for e in entries if lo in e.question.lower() or lo in e.answer.lower()]

    ver_label = version or store.load_manifest().current_version
    table = Table(title=f"Entries — v{ver_label}", box=None, show_edge=False)
    table.add_column("ID", style="cyan bold", width=10)
    table.add_column("Question", max_width=55)
    table.add_column("Answer", max_width=45)
    table.add_column("Tags", style="dim")
    for e in entries:
        table.add_row(e.id, e.question[:55], e.answer[:45], ", ".join(e.tags) or "-")
    console.print(table)
    console.print(f"\n[dim]{len(entries)} entries[/]")


@cli.command()
@click.argument("v1")
@click.argument("v2")
@click.pass_context
def diff(ctx: click.Context, v1: str, v2: str) -> None:
    """Show differences between two committed versions."""
    store: DatasetStore = ctx.obj["store"]
    try:
        result = store.diff(v1, v2)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)

    added, removed, changed = result["added"], result["removed"], result["changed"]
    console.print(
        f"\n[bold]v{v1}[/] -> [bold]v{v2}[/]  "
        f"[green]+{len(added)} added[/]  [red]-{len(removed)} removed[/]  [yellow]~{len(changed)} changed[/]\n"
    )
    for e in added:
        console.print(f"  [green]+[/] [{e.id}] {e.question[:80]}")
    for e in removed:
        console.print(f"  [red]-[/] [{e.id}] {e.question[:80]}")
    for e in changed:
        console.print(f"  [yellow]~[/] [{e.id}] {e.question[:80]}")


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["jsonl", "csv", "json"], case_sensitive=False), default="jsonl", show_default=True)
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--version", "-v", default=None, help="Version to export")
@click.pass_context
def export(ctx: click.Context, fmt: str, output: str | None, version: str | None) -> None:
    """Export a committed version to a file."""
    store: DatasetStore = ctx.obj["store"]
    try:
        manifest = store.load_manifest()
        entries = store.load_version(version)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)

    ver = version or manifest.current_version
    out_path = Path(output or f"{manifest.name}_v{ver}.{fmt}")
    exporters = {"jsonl": export_jsonl, "csv": export_csv, "json": export_json}
    exporters[fmt](entries, out_path)
    console.print(f"[green]OK[/] Exported [bold]{len(entries)}[/] entries to [bold]{out_path}[/]")


@cli.command()
@click.option("--answers-file", "-a", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--output", "-o", default=None)
@click.option("--version", "-v", default=None)
@click.option("--threshold", default=0.7, show_default=True)
@click.pass_context
def evaluate(ctx: click.Context, answers_file: Path | None, output: str | None, version: str | None, threshold: float) -> None:
    """Evaluate LLM answers against golden answers using semantic similarity."""
    store: DatasetStore = ctx.obj["store"]
    try:
        manifest = store.load_manifest()
        entries = store.load_version(version)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)

    if answers_file:
        raw = json.loads(answers_file.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            console.print("[red]Error:[/] --answers-file must contain a JSON array of strings.")
            sys.exit(1)
        answers = raw
    else:
        console.print(f"[bold]Evaluating {len(entries)} entries[/] — enter the actual LLM answer for each:\n")
        answers = []
        for e in entries:
            console.print(f"[cyan bold]Q:[/] {e.question}")
            console.print(f"[dim]Expected:[/] {e.answer}")
            answers.append(click.prompt("Actual answer"))
            console.print()

    if len(answers) != len(entries):
        console.print(f"[red]Error:[/] {len(entries)} entries but {len(answers)} answers provided.")
        sys.exit(1)

    console.print("Computing semantic similarity...")
    evaluator = Evaluator()
    summary = evaluator.evaluate_dataset(
        entries, answers, manifest.name, version or manifest.current_version or "unknown"
    )

    if output:
        path = Path(output)
        path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    else:
        path = store.save_eval(summary)

    avg = summary.avg_semantic_similarity or 0.0
    colour = "green" if avg >= threshold else "red"
    console.print(f"\n[bold]Evaluation Results[/]  (v{summary.version})")
    console.print(f"  Avg Semantic Similarity : [{colour}]{avg:.3f}[/]")
    console.print(f"  Pass (>={threshold})         : [{colour}]{'YES' if avg >= threshold else 'NO'}[/]")
    console.print(f"\n  Results saved -> [bold]{path}[/]")


def main() -> None:
    cli(obj={})
