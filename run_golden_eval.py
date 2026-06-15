"""Run the RAG pipeline against the golden dataset and evaluate results."""

import argparse
import json
import time
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from golden_dataset import DatasetStore, Evaluator

# --- RAG Pipeline Setup ---

loader = TextLoader("data/company_policy.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

def ask_rag(llm, question: str, model_name: str) -> str:
    """Run a question through the RAG pipeline and return the answer using `llm`.

    Adds error handling to give a helpful hint when Ollama rejects a model name.
    """
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""You are an HR assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        msg = str(e)
        print(f"Error invoking LLM '{model_name}': {msg}")
        if "invalid model name" in msg.lower() or "invalid model" in msg.lower():
            print(
                "Hint: the Ollama model name may differ. If your local model is named 'gpt-oss:20b',"
                " try `--llm gpt-oss:20b` and ensure the model is installed in Ollama.`"
            )
        raise


# --- Golden Dataset Evaluation ---

def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline against golden dataset")
    parser.add_argument(
        "--llm",
        default="llama3.2",
        help="LLM model to use for answering (example: llama3.2 or gpt-oss:20b).",
    )
    args = parser.parse_args()

    # instantiate chosen LLM (pass-through model name; Ollama must have it installed)
    selected_model = args.llm
    llm = ChatOllama(model=selected_model, temperature=0)

    store = DatasetStore()
    entries = store.load_version()
    manifest = store.load_manifest()

    print(f"Running RAG pipeline against {len(entries)} golden entries (v{manifest.current_version})...\n")

    answers = []
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] Q: {entry.question}")
        start = time.time()
        answer = ask_rag(llm, entry.question, selected_model)
        latency = round(time.time() - start, 2)
        print(f"         A: {answer}")
        print(f"         Expected: {entry.answer}")
        print(f"         Latency: {latency}s\n")
        answers.append(answer)

    # Save raw answers for future re-evaluation without re-running the LLM
    answers_path = Path("my_answers.json")
    answers_path.write_text(json.dumps(answers, indent=2), encoding="utf-8")
    print(f"Saved LLM answers to {answers_path}\n")

    # Evaluate
    evaluator = Evaluator()
    summary = evaluator.evaluate_dataset(
        entries, answers, manifest.name, manifest.current_version
    )

    # Save eval
    eval_path = store.save_eval(summary)

    # Print results
    avg = summary.avg_semantic_similarity or 0.0
    passed = summary.passed()
    GREEN = "\x1b[32m"
    RED = "\x1b[31m"
    RESET = "\x1b[0m"
    TICK = "✓"
    CROSS = "✗"

    print("=" * 50)
    print(f"EVALUATION RESULTS (v{summary.version})")
    print("=" * 50)
    for r in summary.results:
        symbol = f"{GREEN}{TICK}{RESET}" if (r.semantic_similarity or 0.0) >= 0.5 else f"{RED}{CROSS}{RESET}"
        print(f"  [{r.entry_id}] {symbol} similarity={r.semantic_similarity:.3f}  Q: {r.question[:50]}")
    overall_symbol = f"{GREEN}{TICK}{RESET}" if passed else f"{RED}{CROSS}{RESET}"
    print(f"\n  Avg Similarity: {avg:.3f}")
    print(f"  Pass (>=0.5):   {overall_symbol}")
    print(f"  Saved to:       {eval_path}")


if __name__ == "__main__":
    main()
