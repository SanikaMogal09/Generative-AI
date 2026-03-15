from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
  template='Write a joke about {topic}',
  input_variables=['topic']
)

model = ChatOllama(model="llama3")

parser = StrOutputParser()

prompt2 = PromptTemplate(
  template='Explain the following joke - {text} ',
  input_variables=['text']
)

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))

