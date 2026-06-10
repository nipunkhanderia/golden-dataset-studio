from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

loader = TextLoader("data/context.txt")
doc = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size = 20, chunk_overlap = 2)
chunk = splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(chunk, embeddings)
retriever = db.as_retriever(search={"k":3})


question = "One venus day = how many earth days?"
retrieve = retriever.invoke(question)


content = []
for r in retrieve:
   content.append(r.page_content)

full_content = "\n".join(content)

# print (full_content)

llm = ChatOllama(model = "gpt-oss:20b", temperature=0)

prompt = f"""Give answer form hte context and not your intenral memory parameters.


the context is this {full_content} 

and the questions ins this - {question}"""

# print(prompt)

response = llm.invoke(prompt)

print(response.content)




from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from groq import Groq
from deepeval.metrics import AnswerRelevancyMetric

class Groq_API(DeepEvalBaseLLM):
   def __init__(self):
        self.client = Groq()
    
   def generate(self, prompt):
        response = self.client.chat.completions.create(
           model="llama-3.3-70b-versatile",messages=[{"role":"user", "content":prompt}]
       )
        return response.choices[0].message.content
   
   async def a_generate(self, prompt):
        return self.generate(prompt)
   
   def get_model_name(self):
       return "groq/llama-3.3-70b-versatile"
   def load_model(self):
       return self.client
   
groq = Groq_API()

from deepeval.metrics import FaithfulnessMetric

test = LLMTestCase(actual_output=response.content, retrieval_context=content,
                   expected_output="238 days", input = prompt, context=content)
metric = AnswerRelevancyMetric(threshold=0.5, model = groq)
print(metric.measure(test))
print(metric.is_successful())

faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=groq)


from langfuse import Langfuse

langfuse = Langfuse()

langfuse.create_event(name = "New test added",metadata={
    "Question":question,
    "prompt":prompt,
    "Answer":response,
    "Measure":metric.measure(test),
    "Pass/Fail":metric.is_successful(),
    "Score":metric.score,
    "Faitfulness measure":faithfulness_metric.measure(test),
    "Faithfulness pass/fail":faithfulness_metric.is_successful(),
    "Faithful score":faithfulness_metric.score

}

)


    


   






