from test_cases import test_cases

from langchain_community.document_loaders import TextLoader
from observability.langfuse_setup import langfuse
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama

# Load docs
loader = TextLoader("data/company_policy.txt")
docs = loader.load()

# Split docs
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

# Local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store
db = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = db.as_retriever()

# Local LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

question = "How many annual leave days do employees get?"
expected = "25 days"


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





def run_test(question, expected):

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an HR assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    start_time = time.time()

    response = llm.invoke(prompt)

    latency = round(time.time() - start_time, 2)

    answer = response.content

    passed = expected.lower() in answer.lower()

    print("\n-------------------")
    print("Question:", question)
    print("Expected:", expected)
    print("Actual:", answer)
    print("Result:", "PASS" if passed else "FAIL")

    langfuse.create_event(
        name="evaluation",
        metadata={
            "question": question,
            "expected": expected,
            "actual": answer,
            "passed": passed,
            "latency_seconds": latency
        }
    )



for test_case in test_cases:
    run_test(test_case)





with langfuse.start_as_current_observation(
    name="rag-eval-project"
):
    with langfuse.start_as_current_observation(
    name="rag-eval-project"
):

        print(
            "Trace ID:",
            langfuse.get_current_trace_id()
        )

    # Retrieve documents
    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    print("\nRetrieved Context:")
    print(context)

    # Build prompt
    prompt = f"""
You are an HR assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    print("\nPrompt:")
    print(prompt)

    # Measure latency
    start_time = time.time()

    response = llm.invoke(prompt)

    latency = round(time.time() - start_time, 2)

    answer = response.content

    passed = expected.lower() in answer.lower()

    print("\nTest Result:")
    print("PASS" if passed else "FAIL")


    # Log question
    langfuse.create_event(
        name="question",
        input={
            "question": question
        }
    )

    # Log retrieved context
    langfuse.create_event(
        name="retrieved_context",
        output={
            "context": context
        }
    )

    # Log answer
    langfuse.create_event(
        name="answer",
        output={
            "answer": response.content
        },
        metadata={
            "expected_answer": "25 days",
            "test_type": "retrieval_accuracy",
            "model": "llama3.2",
            "latency_seconds": latency
        }
    )
    langfuse.create_event(
    name="evaluation",
    metadata={
        "passed": passed,
        "expected": expected,
        "actual": answer
    }
    )
    print("Trace ID:", langfuse.get_current_trace_id())
    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(response.content)





with langfuse.start_as_current_observation(
    name="rag-eval-suite"
):
    for tc in test_cases:
        run_test(
            tc["question"],
            tc["expected"]
        )

langfuse.flush()

langfuse.flush()


