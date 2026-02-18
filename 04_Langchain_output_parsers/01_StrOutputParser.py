from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.invoke({'topic':'Black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)