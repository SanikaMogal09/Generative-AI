# from langchain_community.document_loaders import TextLoader

# loader = TextLoader('01_cricket.txt', encoding='utf-8')

# docs = loader.load()

# # print(docs)
# print(type(docs))
# print(len(docs))
# print(docs[0])
# print(docs[0].page_content)
# print(docs[0].metadata)

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatOllama(model="llama3")

prompt = PromptTemplate(
  template="Write a summary for the following poem - \n{poem}",
  input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('01_cricket.txt',encoding="utf-8")

docs = loader.load()

# print(docs)
# print(type(docs))
# print(len(docs))
# print(docs[0])

chain = prompt| model | parser

result = chain.invoke({'poem':docs[0].page_content})

print(result)