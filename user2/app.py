

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


loader = TextLoader("data/context.txt")
doc = loader.load()

# print(doc)

split = RecursiveCharacterTextSplitter(chunk_size = 20, chunk_overlap=2)
chucnks = split.split_documents(doc)


space = []
for chucnk in chucnks:
    bread_crumbs = chucnk.page_content
    space.append(bread_crumbs)
print(len(space))


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2" )


print(embedding)


db = FAISS.from_documents(chucnks, embedding)

print(db)


