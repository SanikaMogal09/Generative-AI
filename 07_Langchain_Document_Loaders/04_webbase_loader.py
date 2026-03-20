# from langchain_community.document_loaders import WebBaseLoader

# url = 'https://www.selenium.dev/documentation/test_practices/overview/'

# loader = WebBaseLoader(url)

# docs = loader.load()

# print(docs[0].page_content)
# print(len(docs))

from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

model = ChatOllama(model="llama3")

prompt = PromptTemplate(
  template='Answer the following question \n {question} from the following text -\n {text}',
  input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.selenium.dev/documentation/test_practices/overview/'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt| model| parser

result = chain.invoke({'question':'What is the main advantage of Selenium tests compared to other testing methods?' , 'text':docs[0].page_content})
print(result)

