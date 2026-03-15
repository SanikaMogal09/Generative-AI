# from langchain_core.runnables import RunnableLambda

# def word_counter(text):
#   return len(text.split())

# words = RunnableLambda(word_counter)

# print(words.invoke('Hi there how are you ?'))

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
  template='Generate a joke on {topic}',
  input_variables=['topic']
)

model = ChatOllama(model='llama3')

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt,model,parser)

def word_counter(text):
  return len(text.split())

parallel_chain = RunnableParallel({
  'joke':RunnablePassthrough(),
  'count':RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'Football'})

# print(result)

final_result = """ {}\n word count - {}""".format(result['joke'],result['count'])

print(final_result)