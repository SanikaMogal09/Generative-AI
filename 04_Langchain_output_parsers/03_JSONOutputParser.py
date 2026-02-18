from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatOllama(model="llama3")

parser = JsonOutputParser()


# parser.get_format_instructions() generates automatic instructions that tell the LLM: “You MUST return the output in valid JSON format.”
template = PromptTemplate(
  # template='Give me the name, age and city of a fictional person \n {format_instruction}',
  template='Give me 5 facts about {topic} \n {format_instruction}',
  input_variables=['topic'],
  partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# # print(result)

# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))

chain = template | model | parser

result = chain.invoke({'topic':'blackhole'})

print(result)