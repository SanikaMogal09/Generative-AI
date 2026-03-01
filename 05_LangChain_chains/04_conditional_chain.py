from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

# we use pydantic output parser for structuring the output given by classifier chain as positive or negative

class Feedback(BaseModel):
  sentiment: Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
  template= 'Classify the sentiment of the following feedback text into positive or negative \n{feedback} \n {format_instruction}',
  input_variables=['feedback'],
  partial_variables={'format_instruction': parser2.get_format_instructions()}
)




classifier_chain = prompt | model | parser2

# print(classifier_chain.invoke({'feedback':'This is a terrible smartphone'}))

# result = classifier_chain.invoke({'feedback':'This is a terrible smartphone'}).sentiment

# print(result)

prompt1 = PromptTemplate(
  template='Write an appropriate resopnse to this positive feedback \n {feedback}',
  input_variables=['feedback']
)

prompt2 = PromptTemplate(
  template='Write an appropriate resopnse to this negative feedback \n {feedback}',
  input_variables=['feedback']
)

branch_chain = RunnableBranch(
  (lambda x: x["sentiment"] == 'positive'
 , prompt1 | model | parser),             # (condition1, chain)
  (lambda x: x["sentiment"] == 'negative'
, prompt2 | model | parser),              # (condition2,chain)
  RunnableLambda(lambda x : "could not find sentiment")                        # default chain
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'This is a terrible smartphone'})

print(result)

chain.get_graph().print_ascii()


