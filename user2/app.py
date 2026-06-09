

from langchain_community.document_loaders import TextLoader


loader = TextLoader("data/context.txt")
doc = loader.load()

print(doc)