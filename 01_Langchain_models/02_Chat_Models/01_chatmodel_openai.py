# https://platform.openai.com/
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini', temperature=1.5, max_completion_tokens=20)

result = model.invoke("Write a five line poem on Tree.")

print(result)