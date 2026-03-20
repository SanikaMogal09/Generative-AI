from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
  path='03_books',
  glob='**/*.pdf',
  loader_cls=PyPDFLoader
)

# docs = loader.load()

# print(len(docs))
# print(docs[8].page_content)
# print(docs[8].metadata)


# docs = loader.load()

# for document in docs:
#   print(document.metadata)


docs = loader.lazy_load()

for document in docs:
  print(document.metadata)