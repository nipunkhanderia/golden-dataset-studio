

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


loader = TextLoader("data/context.txt")
doc = loader.load()

# print(doc)

split = RecursiveCharacterTextSplitter(chunk_size = 20, chunk_overlap=2)
chucnks = split.split_documents(doc)


space = []
for chucnk in chucnks:
    bread_crumbs = chucnk.page_content
    space.append(bread_crumbs)
# print(len(space))


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2" )


# print(embedding)


db = FAISS.from_documents(chucnks, embedding)

# print(db)


llm = ChatOllama(model="gpt-oss:20b", temperature =0)

# response = llm.invoke("Why is sky blue, tell me in one sentence only.")

# print(response.content)


retriever = db.as_retriever(search_kwargs = {"k":3})


# print(retriever)

question = "Who is Niels Armstrong?"
retrieve = retriever.invoke(question)
p_retrive = []
for r in retrieve:
    content_p = r.page_content
    p_retrive.append(content_p)
    final_context = "\n".join(p_retrive)
# print(final_context)
prompt = f"Answer the question from the context only and not your memomry - Following is the question {question} and this is the context {final_context} "

reposnse = llm.invoke(prompt)
# print(reposnse.content)

from deepeval.test_case import LLMTestCase
from groq import Groq
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import DeepEvalBaseLLM

class GroqAPI(DeepEvalBaseLLM):
    def __init__(self):
        self.client = Groq()
    
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages= [{"role":"user", "content":prompt}]
        )
        return response.choices[0].message.content
    async def a_generate(self,prompt):
        return self.generate(prompt)
    
    def get_model_name(self):
        return "groq/llama-3.3-70b-versatile"
    def load_model(self):
        return self.client

groq = GroqAPI()
test = LLMTestCase(input=prompt, actual_output=reposnse.content, 
                   expected_output="First man on moon", context=p_retrive)
metric = AnswerRelevancyMetric(threshold=0.5, model=groq)
print(test)
print(metric.measure(test))
print(metric.score)
print(metric.is_successful())


from langfuse  import Langfuse

langfuse = Langfuse()


langfuse.create_event(name="user tests", metadata={
    "question":question,
    "expected":"First man on moon",
    "Score":metric.score,
    "Did it pass":metric.is_successful()
})








