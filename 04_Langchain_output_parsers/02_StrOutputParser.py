from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOllama(model="llama3")


# prompt 1 -> detailed Report 

template1 = PromptTemplate(
  template='Write a detailed report on {topic}',
  input_variables = ['topic']
)

# prompt 2 -> 5 lines summary 
template2 = PromptTemplate(
  template='Write a write a 5 lines summary on the following text. /n{text}',
  input_variables = ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)