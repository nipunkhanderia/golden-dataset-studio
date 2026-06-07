# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import ChatOllama

# loader = TextLoader("data/nipun_policy.txt")

# document = loader.load()

# # print(document)

# chuncks = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
# split = chuncks.split_documents(document)
# # print(split)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # print(f"embessings is {embeddings}")

# db = FAISS.from_documents(split, embeddings)
# # print(f"db is {db}")

# retriever = db.as_retriever(search_kwargs={"k":3})

# # print(retriever)

# llm = ChatOllama(
#     model="llama3.2:latest",
#     temperature="0"
# )

# response = llm.invoke("Tell me history of India in 3 sentences")

# # print(llm)

# print(response)





from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_ollama import ChatOllama

loader = TextLoader("data/nipun_policy.txt")
doc = loader.load()

split = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
chuncks = split.split_documents(doc)

print(chuncks)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(chuncks, embeddings)

retriever = db.as_retriever(search_kwargs = {"k":3})


llm = ChatOllama(model="gpt-oss:20b", temperature="0")

prompt = "who is nipun khanderia?"
response = llm.invoke(prompt)












