# # from langchain_community.document_loaders import TextLoader
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_ollama import ChatOllama

# # loader = TextLoader("data/nipun_policy.txt")

# # document = loader.load()

# # # print(document)

# # chuncks = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
# # split = chuncks.split_documents(document)
# # # print(split)

# # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # # print(f"embessings is {embeddings}")

# # db = FAISS.from_documents(split, embeddings)
# # # print(f"db is {db}")

# # retriever = db.as_retriever(search_kwargs={"k":3})

# # # print(retriever)

# # llm = ChatOllama(
# #     model="llama3.2:latest",
# #     temperature="0"
# # )

# # response = llm.invoke("Tell me history of India in 3 sentences")

# # # print(llm)

# # print(response)





# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS 
# from langchain_ollama import ChatOllama

# from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.test_case import LLMTestCase

# loader = TextLoader("data/nipun_policy.txt")
# doc = loader.load()

# split = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
# chuncks = split.split_documents(doc)

# # print(chuncks)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# db = FAISS.from_documents(chuncks, embeddings)

# retriever = db.as_retriever(search_kwargs = {"k":3})
# # print(retriev)

# llm = ChatOllama(model="gpt-oss:20b", temperature="0")

# # prompt = "who is nipun khanderia?"
# # response = llm.invoke(prompt)
# # print(response)



# from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.test_case import LLMTestCase
# from deepeval.models import DeepEvalBaseLLM
# from groq import Groq

# class GroqModel(DeepEvalBaseLLM):
#     def __init__(self):
#         self.client = Groq()  # picks up GROQ_API_KEY from .env automatically

#     def load_model(self):
#         return self.client

#     def generate(self, prompt: str) -> str:
#         response = self.client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response.choices[0].message.content

#     async def a_generate(self, prompt: str) -> str:
#         return self.generate(prompt)

#     def get_model_name(self):
#         return "groq/llama-3.3-70b-versatile"
    


# def test_run(question, expected):
#     retriev = retriever.invoke("Who is Nipun?")
#     contexts = []
#     for doc in retriev:
#         contexts.append(doc.page_content)

   
#     print(contexts)
#     context = "\n".join(contexts)
#     print(context)
    
#     # context = "\n".join(contexts)
#     # print(context)
#     prompt = f"Answer the below question from the context. {question} and the context is {context}"

#     response = llm.invoke(prompt)
#     print(response)
#     answer = response.content.strip()
#     print(answer)


    






#     # eval = evaluate_response(
#     #     question=question,
#     #     answer=answer,
#     #     expected=expected,
#     #     contexts=context
#     # )
#     # print(eval)

#     # metric = AnswerRelevancyMetric(threshold=0.5)
#     # metric.measure(test_case)

#     # print(f"Score  : {metric.score}")
#     # print(f"Passed : {metric.is_successful()}")
#     # print(f"Reason : {metric.reason}")


# groq_model = GroqModel()
# metric = AnswerRelevancyMetric(threshold=0.5, model=groq_model)
# metric.measure(test_case)

# test_case = LLMTestCase(
#         input=question,
#         actual_output=answer,
#         expected_output=expected,
#         retrieval_context=contexts
# )


# test_run("Who is Nipun", "Nipun is software engineer")



from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from groq import Groq
from langfuse import Langfuse


# --- setup ---
loader = TextLoader("data/nipun_policy.txt")
doc = loader.load()

split = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = split.split_documents(doc)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="gpt-oss:20b", temperature="0")

# --- groq judge model ---
class GroqModel(DeepEvalBaseLLM):
    def __init__(self):
        self.client = Groq()

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "groq/llama-3.3-70b-versatile"

langfuse = Langfuse()

# --- test function ---
def test_run(question, expected):
    retriev = retriever.invoke("Who is Nipun?")
    contexts = []
    for doc in retriev:
        contexts.append(doc.page_content)

    context = "\n".join(contexts)

    prompt = f"Answer the below question from the context. {question} and the context is {context}"
    response = llm.invoke(prompt)
    answer = response.content.strip()
    print(answer)


    test_case = LLMTestCase(input = question, expected_output=expected, retrieval_context=contexts, actual_output=answer)
    groq = GroqModel()
    metric = AnswerRelevancyMetric(threshold=0.5, model = groq)
    metric1 = FaithfulnessMetric(threshold=0.5, include_reason=True, model = groq)
    metric.measure(test_case)
    metric.is_successful()
    metric1.measure(test_case)


    print(f"Score : {metric.score}")
    print(f"Passed :{metric.is_successful()}")
    print(f"Reason : {metric.reason}")
    print(f"Faithfulness Score:{metric1.score}")
    print(f"Faithfulness PAssed:{metric1.is_successful()}")
    print(f"Faithfulness Reason:{metric1.reason}")



    langfuse.create_event(
            name="evaluation",
            metadata={
                "question":            question,
                "expected":            expected,
                "actual":              answer,
                "relevancy_score":     metric.score,
                "relevancy_passed":    metric.is_successful(),
                "relevancy_reason":    metric.reason,
                "faithfulness_score":  metric1.score,
                "faithfulness_passed": metric1.is_successful(),
                "faithfulness_reason": metric1.reason
            }
        )
    


    # test_case = LLMTestCase(
    #     input=question,
    #     actual_output=answer,
    #     expected_output=expected,
    #     retrieval_context=contexts
    # )

    # groq_model = GroqModel()
    # metric = AnswerRelevancyMetric(threshold=0.5, model=groq_model)
    # metric.measure(test_case)

    # print(f"Score  : {metric.score}")
    # print(f"Passed : {metric.is_successful()}")
    # print(f"Reason : {metric.reason}")


# --- run ---
test_run("Who is Nipun", "Nipun is software engineer")













