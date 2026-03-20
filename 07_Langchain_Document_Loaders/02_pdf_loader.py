from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("02_Atomic_habits.pdf")

docs = loader.load()

# print(docs)
print(len(docs))

print(docs[8].page_content)
print(docs[0].metadata)