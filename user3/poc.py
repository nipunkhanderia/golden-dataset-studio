

from langchain_ollama import ChatOllama

llm = ChatOllama(model="gpt-oss:20b",temperature=0)

response = llm.invoke("Why is sky blue, tell me in one sentence")
print(response.content)