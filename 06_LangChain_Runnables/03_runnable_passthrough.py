from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# passthrough = RunnablePassthrough()

# print(passthrough.invoke({'name':'Sanika'}))

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

joke_gene_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
  'joke':RunnablePassthrough(),
  'explanation': RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gene_chain,parallel_chain)

result = final_chain.invoke({'topic':'cricket'})

print(result)