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




