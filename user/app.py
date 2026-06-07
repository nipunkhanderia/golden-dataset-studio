# ============================================================
# WHAT THIS FILE DOES (big picture)
# ============================================================
# This is a RAG (Retrieval-Augmented Generation) evaluation pipeline.
#
# In plain English:
#   1. Load a text document (company policy)
#   2. Break it into small chunks and store them as vectors (numbers)
#   3. For each test question, find the most relevant chunks
#   4. Send those chunks + the question to an LLM (Ollama / llama3.2)
#   5. Score the LLM's answer using DeepEval (an AI testing library)
#   6. Log everything to Langfuse (an observability/monitoring tool)
#
# Why does this matter for AI testing jobs?
#   - RAG is one of the most common patterns in production AI apps
#   - Evaluating RAG quality is a core skill for AI QA/test engineers
#   - DeepEval + Langfuse are real tools used in industry
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

# TextLoader: reads a plain .txt file from your disk
from langchain_community.document_loaders import TextLoader

# RecursiveCharacterTextSplitter: breaks a big document into smaller pieces
# (LLMs can only handle limited amounts of text at once — called "context window")
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS: a vector database (stores text as numbers so we can search by meaning)
from langchain_community.vectorstores import FAISS

# HuggingFaceEmbeddings: converts text into vectors (lists of numbers)
# The model "all-MiniLM-L6-v2" is free and runs locally on your machine
from langchain_huggingface import HuggingFaceEmbeddings

# ChatOllama: lets you talk to a locally-running LLM (Ollama must be installed)
from langchain_ollama import ChatOllama

# Our own helper files (you need to create these too in your project)
# langfuse_setup.py exports a `langfuse` client for logging
from observability.langfuse_setup import langfuse

# deepeval_eval.py exports `evaluate_response` — our custom scoring function
from evaluation.deepeval_eval import evaluate_response

# time: built-in Python module to measure how long things take (latency)
import time


# ============================================================
# STEP 1: LOAD THE DOCUMENT
# ============================================================

# TextLoader reads the file and returns a list of "Document" objects
# Each Document has .page_content (the text) and .metadata (file info)
loader = TextLoader("data/company_policy.txt")
docs = loader.load()  # docs is a list — usually just 1 item for a single file


# ============================================================
# STEP 2: SPLIT INTO CHUNKS
# ============================================================

# Why split? LLMs work better with small, focused pieces of text.
# chunk_size=200    → each chunk is max 200 characters
# chunk_overlap=20  → each chunk shares 20 characters with the next one
#                     (overlap helps avoid losing context at the boundaries)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

# split_documents returns a list of smaller Document objects
chunks = splitter.split_documents(docs)


# ============================================================
# STEP 3: CREATE EMBEDDINGS AND STORE IN FAISS
# ============================================================

# An "embedding" turns text into a list of numbers (a vector).
# Similar texts get similar vectors — so we can search by meaning, not just keywords.
#
# "sentence-transformers/all-MiniLM-L6-v2" is a small, fast model that runs locally.
# It's free and very popular for this kind of task.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS.from_documents() takes all our chunks, turns them into vectors,
# and stores everything in memory so we can search later.
db = FAISS.from_documents(
    chunks,      # the small text pieces
    embeddings   # the model that converts text → vectors
)

# A "retriever" is an object that can answer: "given a question,
# which chunks are most relevant?"
# search_kwargs={"k": 3} means → return the top 3 most relevant chunks
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# STEP 4: SET UP THE LLM
# ============================================================

# ChatOllama connects to Ollama running on your local machine.
# You need to have Ollama installed and run: ollama pull llama3.2
#
# temperature=0.0 means: be deterministic (always give the same answer).
# Higher temperature = more creative / random answers.
llm = ChatOllama(
    model="llama3.2",
    temperature=0.0
)


# ============================================================
# STEP 5: DEFINE TEST CASES
# ============================================================

# Each test case has:
#   "question" → what we ask the LLM
#   "expected" → the correct answer we expect (used for scoring)
#
# The last case tests what happens when the answer is NOT in the document.
# A good RAG system should say "I don't know" rather than making something up.
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
        "expected": "I don't know"   # ← tests hallucination: is this in the doc?
    }
]


# ============================================================
# STEP 6: THE TEST RUNNER FUNCTION
# ============================================================

def run_test(question, expected):
    """
    Runs a single test case end-to-end:
      1. Retrieves relevant chunks from the vector store
      2. Builds a prompt (context + question)
      3. Gets an answer from the LLM
      4. Scores the answer with DeepEval
      5. Logs everything to Langfuse
      6. Returns a result dictionary
    """

    # --- 6a. Retrieve relevant chunks ---
    # retriever.invoke() searches FAISS and returns the top 3 matching chunks
    retrieved_docs = retriever.invoke(question)

    # Build two representations of the retrieved content:
    #   contexts → a Python list, each item is one chunk's text
    #             (DeepEval needs this as a list)
    #   context  → a single string joining all chunks with newlines
    #             (the LLM prompt needs this as plain text)
    contexts = [doc.page_content for doc in retrieved_docs]
    context = "\n".join(contexts)

    # --- 6b. Build the prompt ---
    # This is a "RAG prompt": we tell the LLM to ONLY use the provided context.
    # The instruction "I don't know" handles unanswerable questions.
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

    # --- 6c. Call the LLM and measure latency ---
    # time.time() gives the current time in seconds (as a float)
    start_time = time.time()

    # llm.invoke() sends the prompt to Ollama and returns a response object
    response = llm.invoke(prompt)

    # Calculate how long the LLM took to respond, rounded to 2 decimal places
    latency = round(time.time() - start_time, 2)

    # .content holds the actual text of the LLM's reply
    # .strip() removes any leading/trailing whitespace
    answer = response.content.strip()

    # --- 6d. Score with DeepEval ---
    # evaluate_response() is our custom function (in evaluation/deepeval_eval.py)
    # It runs multiple DeepEval metrics and returns a dictionary of scores.
    eval_result = evaluate_response(
        question=question,   # the original question
        answer=answer,       # what the LLM actually said
        expected=expected,   # what we expected it to say
        contexts=contexts    # the chunks we retrieved (needed for RAG metrics)
    )

    # Pull out the summary fields from the evaluation result
    score   = eval_result["overall_score"]  # a float like 0.85
    passed  = eval_result["overall_pass"]   # True or False
    reason  = str(eval_result["metrics"])   # per-metric breakdown as a string

    # --- 6e. Print results to the terminal ---
    print("\n" + "=" * 60)
    print("Question :", question)
    print("Expected :", expected)
    print("Actual   :", answer)
    print("Latency  :", latency, "seconds")
    print("\nDeepEval")
    print("Score    :", score)
    print("Passed   :", passed)
    print("Reason   :", reason)
    print("Result   :", "PASS" if passed else "FAIL")

    # --- 6f. Log to Langfuse ---
    # Langfuse is an observability tool — it records every evaluation run
    # so you can review results in a dashboard later.
    # We wrap this in try/except so a Langfuse error doesn't crash the whole test.
    try:
        langfuse.create_event(
            name="evaluation",
            metadata={
                "question":        question,
                "expected":        expected,
                "actual":          answer,
                "passed":          passed,
                "deepeval_score":  score,
                "deepeval_reason": reason,
                "latency_seconds": latency
            }
        )
    except Exception as e:
        # If Langfuse isn't running or credentials are wrong, just warn and continue
        print(f"Langfuse logging warning: {e}")

    # Return a dict so we can aggregate results later
    return {
        "question": question,
        "expected": expected,
        "actual":   answer,
        "passed":   passed,
        "score":    score,
        "reason":   reason,
        "latency":  latency
    }


# ============================================================
# STEP 7: RUN THE FULL EVALUATION SUITE
# ============================================================

results = []  # we'll collect each test result here

# langfuse.start_as_current_observation() groups all test runs under one "trace"
# in Langfuse — useful for seeing the whole suite in one place on the dashboard
with langfuse.start_as_current_observation(name="rag-eval-suite"):

    # Try to print the trace ID (useful for finding this run in the Langfuse UI)
    try:
        print("Trace ID:", langfuse.get_current_trace_id())
    except Exception:
        pass  # If it fails, just keep going

    # Loop through each test case and run it
    for tc in test_cases:
        result = run_test(tc["question"], tc["expected"])
        results.append(result)


# ============================================================
# STEP 8: PRINT SUMMARY
# ============================================================

# Count how many tests passed
passed_count = sum(1 for r in results if r["passed"])
total_count  = len(results)

# Calculate the average DeepEval score across all tests
avg_score = round(
    sum(r["score"] for r in results) / total_count,
    3
)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"Passed       : {passed_count}/{total_count}")
print(f"Success Rate : {(passed_count / total_count) * 100:.1f}%")
print(f"Avg Score    : {avg_score}")


# ============================================================
# STEP 9: FLUSH LANGFUSE
# ============================================================

# Langfuse sends data in batches — flush() forces it to send
# any remaining logs before the script exits
try:
    langfuse.flush()
except Exception as e:
    print(f"Langfuse flush warning: {e}")