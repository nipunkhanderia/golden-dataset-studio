# # from evaluation.ragas_eval import evaluate_response
# from langchain_community.document_loaders import TextLoader
# from observability.langfuse_setup import langfuse
# import time
# from evaluation.deepeval_eval import evaluate_response

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama

# # ============================================================
# # Load documents
# # ============================================================

# loader = TextLoader("data/company_policy.txt")
# docs = loader.load()

# # ============================================================
# # Split documents
# # ============================================================

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=20
# )

# chunks = splitter.split_documents(docs)

# # ============================================================
# # Embeddings + Vector Store
# # ============================================================

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# db = FAISS.from_documents(
#     chunks,
#     embeddings
# )

# retriever = db.as_retriever()

# # ============================================================
# # LLM
# # ============================================================

# llm = ChatOllama(
#     model="llama3.2",
#     temperature=0
# )

# # ============================================================
# # Test Cases
# # ============================================================

# test_cases = [
#     {
#         "question": "How many annual leave days do employees get?",
#         "expected": "25 days"
#     },
#     {
#         "question": "What is the remote work policy?",
#         "expected": "3 days per week"
#     },
#     {
#         "question": "Who is covered by medical insurance?",
#         "expected": "employees and dependents"
#     },
#     {
#         "question": "What is the maternity leave policy?",
#         "expected": "I don't know"
#     }
# ]

# # ============================================================
# # Test Runner
# # ============================================================

# def run_test(question, expected):

#     # Retrieve documents
#     docs = retriever.invoke(question)

#     context = "\n".join(
#         doc.page_content for doc in docs
#     )


#     contexts = [
#     doc.page_content
#     for doc in docs
#     ]

#     prompt = f"""
# You are an HR assistant.

# Use the context below to answer the question.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     start_time = time.time()

#     response = llm.invoke(prompt)

#     latency = round(
#         time.time() - start_time,
#         2
#     )

#     answer = response.content.strip()


#     passed = expected.lower() in answer.lower()

#     print("\n" + "=" * 60)
#     print("Question :", question)
#     print("Expected :", expected)
#     print("Actual   :", answer)
#     print("Latency  :", latency, "seconds")
#     print("Result   :", "PASS" if passed else "FAIL")

#     # Log evaluation
#     langfuse.create_event(
#         name="evaluation",
#         metadata={
#             "question": question,
#             "expected": expected,
#             "actual": answer,
#             "passed": passed,
#             "latency_seconds": latency
#             }
#         )

#     return {
#         "question": question,
#         "expected": expected,
#         "actual": answer,
#         "passed": passed,
#         "latency": latency
#     }

# # ============================================================
# # Evaluation Suite
# # ============================================================

# with langfuse.start_as_current_observation(
#     name="rag-eval-suite"
# ):

#     print(
#         "Trace ID:",
#         langfuse.get_current_trace_id()
#     )

#     results = []

#     for tc in test_cases:

#         result = run_test(
#             tc["question"],
#             tc["expected"]
#         )

#         results.append(result)

# # ============================================================
# # Summary
# # ============================================================

# passed_count = sum(
#     1 for r in results if r["passed"]
# )

# total_count = len(results)

# print("\n" + "=" * 60)
# print("EVALUATION SUMMARY")
# print("=" * 60)
# print(f"Passed: {passed_count}/{total_count}")
# print(
#     f"Success Rate: {(passed_count / total_count) * 100:.1f}%"
# )

# # ============================================================
# # Flush Langfuse
# # ============================================================

# langfuse.flush()



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

retriever = db.as_retriever()

# ============================================================

# Ollama LLM

# ============================================================

llm = ChatOllama(
model="llama3.2",
temperature=0
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

```
# Retrieve Documents

docs = retriever.invoke(question)

context = "\n".join(
    doc.page_content
    for doc in docs
)

contexts = [
    doc.page_content
    for doc in docs
]

# Build Prompt

prompt = f"""
```

You are an HR assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
respond with:
"I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

```
# Measure Latency

start_time = time.time()

response = llm.invoke(prompt)

latency = round(
    time.time() - start_time,
    2
)

answer = response.content.strip()

# ========================================================
# DeepEval Evaluation
# ========================================================

eval_result = evaluate_response(
    question=question,
    answer=answer,
    expected=expected
)

score = eval_result["score"]
passed = eval_result["passed"]
reason = eval_result["reason"]

# ========================================================
# Output
# ========================================================

print("\n" + "=" * 60)

print("Question :", question)
print("Expected :", expected)
print("Actual   :", answer)

print("Latency  :", latency, "seconds")

print("\nDeepEval")
print("Score    :", score)
print("Passed   :", passed)
print("Reason   :", reason)

print(
    "Result   :",
    "PASS" if passed else "FAIL"
)

# ========================================================
# Langfuse Logging
# ========================================================

langfuse.create_event(
    name="evaluation",
    metadata={
        "question": question,
        "expected": expected,
        "actual": answer,
        "passed": passed,
        "deepeval_score": score,
        "deepeval_reason": reason,
        "latency_seconds": latency
    }
)

return {
    "question": question,
    "expected": expected,
    "actual": answer,
    "passed": passed,
    "score": score,
    "reason": reason,
    "latency": latency
}
```

# ============================================================

# Evaluation Suite

# ============================================================

results = []

with langfuse.start_as_current_observation(
name="rag-eval-suite"
):

```
print(
    "Trace ID:",
    langfuse.get_current_trace_id()
)

for tc in test_cases:

    result = run_test(
        tc["question"],
        tc["expected"]
    )

    results.append(result)
```

# ============================================================

# Summary

# ============================================================

passed_count = sum(
1
for r in results
if r["passed"]
)

total_count = len(results)

avg_score = round(
sum(
r["score"]
for r in results
) / total_count,
3
)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

print(
f"Passed: {passed_count}/{total_count}"
)

print(
f"Success Rate: {(passed_count / total_count) * 100:.1f}%"
)

print(
f"Average DeepEval Score: {avg_score}"
)

# ============================================================

# Flush Langfuse

# ============================================================

langfuse.flush()
