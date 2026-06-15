from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from observability.langfuse_setup import langfuse
from evaluation.deepeval_eval import evaluate_response

import time

# ============================================================
# Load Documents
# ============================================================

loader = TextLoader("data/company_policy.txt")
docs = loader.load()

# ============================================================
# Split Documents
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

# ============================================================
# Embeddings + Vector Store
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# ============================================================
# Ollama LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0.0
)

# ============================================================
# Test Cases
# ============================================================

test_cases = [
    {
        "question": "How many annual leave days do employees get?",
        "expected": "25 days"
    },
    {
        "question": "What is the remote work policy?",
        "expected": "3 days per week"
    },
    {
        "question": "Who is covered by medical insurance?",
        "expected": "employees and dependents"
    },
    {
        "question": "What is the maternity leave policy?",
        "expected": "I don't know"
    }
]

# ============================================================
# Test Runner
# ============================================================

def run_test(question, expected):

    # --------------------------------------------------------
    # Retrieve Documents
    # --------------------------------------------------------

    retrieved_docs = retriever.invoke(question)

    context = "\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    contexts = [
        doc.page_content
        for doc in retrieved_docs
    ]

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an HR assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
respond exactly with:

I don't know.

Context:
{context}

Question:
{question}

Answer:
"""

    # --------------------------------------------------------
    # LLM Call + Latency
    # --------------------------------------------------------

    start_time = time.time()

    response = llm.invoke(prompt)

    latency = round(
        time.time() - start_time,
        2
    )

    answer = response.content.strip()

    # --------------------------------------------------------
    # DeepEval
    # --------------------------------------------------------

    eval_result = evaluate_response(
        question=question,
        answer=answer,
        expected=expected,
        contexts=contexts
    )

    overall_score = eval_result["overall_score"]
    overall_pass = eval_result["overall_pass"]

    metrics = eval_result["metrics"]

    answer_relevancy = metrics["AnswerRelevancyMetric"]
    faithfulness = metrics["FaithfulnessMetric"]
    contextual_precision = metrics["ContextualPrecisionMetric"]
    contextual_recall = metrics["ContextualRecallMetric"]

    # --------------------------------------------------------
    # Console Output
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print("Question :", question)
    print("Expected :", expected)
    print("Actual   :", answer)

    print("Latency  :", latency, "seconds")

    print("\nDeepEval Metrics")
    print("-" * 40)

    for metric_name, metric_data in metrics.items():

        print(f"\n{metric_name}")
        print("Score  :", round(metric_data["score"], 3))
        print("Passed :", metric_data["passed"])
        print("Reason :", metric_data["reason"])

    print("\nOverall Score :", overall_score)
    print("Overall Pass  :", overall_pass)

    print(
        "\nResult :",
        "PASS" if overall_pass else "FAIL"
    )

    # --------------------------------------------------------
    # Langfuse Logging
    # --------------------------------------------------------

    try:

        # Event metadata
        langfuse.create_event(
            name="rag_evaluation",
            metadata={
                "question": question,
                "expected": expected,
                "actual": answer,
                "latency_seconds": latency,

                "overall_score": overall_score,
                "overall_pass": overall_pass,

                "answer_relevancy_score":
                    answer_relevancy["score"],
                "answer_relevancy_reason":
                    answer_relevancy["reason"],

                "faithfulness_score":
                    faithfulness["score"],
                "faithfulness_reason":
                    faithfulness["reason"],

                "contextual_precision_score":
                    contextual_precision["score"],
                "contextual_precision_reason":
                    contextual_precision["reason"],

                "contextual_recall_score":
                    contextual_recall["score"],
                "contextual_recall_reason":
                    contextual_recall["reason"]
            }
        )

        # Optional: Langfuse Scores
        try:

            langfuse.create_score(
                name="answer_relevancy",
                value=answer_relevancy["score"]
            )

            langfuse.create_score(
                name="faithfulness",
                value=faithfulness["score"]
            )

            langfuse.create_score(
                name="contextual_precision",
                value=contextual_precision["score"]
            )

            langfuse.create_score(
                name="contextual_recall",
                value=contextual_recall["score"]
            )

            langfuse.create_score(
                name="overall_score",
                value=overall_score
            )

        except Exception:
            pass

    except Exception as e:
        print(f"Langfuse logging warning: {e}")

    return {
        "question": question,
        "expected": expected,
        "actual": answer,
        "latency": latency,
        "overall_score": overall_score,
        "overall_pass": overall_pass,
        "metrics": metrics
    }


# ============================================================
# Evaluation Suite
# ============================================================

results = []

with langfuse.start_as_current_observation(
    name="rag-eval-suite"
):

    try:
        print(
            "Trace ID:",
            langfuse.get_current_trace_id()
        )
    except Exception:
        pass

    for tc in test_cases:

        result = run_test(
            tc["question"],
            tc["expected"]
        )

        results.append(result)

# ============================================================
# Summary
# ============================================================

passed_count = sum(
    1
    for r in results
    if r["overall_pass"]
)

total_count = len(results)

avg_score = round(
    sum(
        r["overall_score"]
        for r in results
    ) / total_count,
    3
)

print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print(
    f"Passed: {passed_count}/{total_count}"
)

print(
    f"Success Rate: {(passed_count / total_count) * 100:.1f}%"
)

print(
    f"Average Score: {avg_score}"
)

# ============================================================
# Flush Langfuse
# ============================================================

try:
    langfuse.flush()
except Exception as e:
    print(
        f"Langfuse flush warning: {e}"
    )