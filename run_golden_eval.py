"""Run the RAG pipeline against the golden dataset and evaluate results."""

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

llm = ChatOllama(model="llama3.2", temperature=0)


def ask_rag(question: str) -> str:
    """Run a question through the RAG pipeline and return the answer."""
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
    response = llm.invoke(prompt)
    return response.content


# --- Golden Dataset Evaluation ---

def main():
    store = DatasetStore()
    entries = store.load_version()
    manifest = store.load_manifest()

    print(f"Running RAG pipeline against {len(entries)} golden entries (v{manifest.current_version})...\n")

    answers = []
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] Q: {entry.question}")
        start = time.time()
        answer = ask_rag(entry.question)
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
    print("=" * 50)
    print(f"EVALUATION RESULTS (v{summary.version})")
    print("=" * 50)
    for r in summary.results:
        print(f"  [{r.entry_id}] similarity={r.semantic_similarity:.3f}  Q: {r.question[:50]}")
    print(f"\n  Avg Similarity: {avg:.3f}")
    print(f"  Pass (>=0.5):   {'YES' if passed else 'NO'}")
    print(f"  Saved to:       {eval_path}")


if __name__ == "__main__":
    main()
