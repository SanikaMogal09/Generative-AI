# from langchain_text_splitters import CharacterTextSplitter

# text = """

# Space exploration is the scientific study and discovery of outer space using telescopes, satellites, and spacecraft. It has expanded our understanding of the universe, from studying distant galaxies to exploring planets like Mars and the Moon. 

# Missions conducted by organizations like NASA and ISRO have led to important advancements in technology, communication, and our knowledge of Earth's place in the cosmos. Space exploration also inspires innovation and curiosity, pushing humanity to discover new possibilities beyond our planet.

# """
# splitter = CharacterTextSplitter(
#   separator='',
#   chunk_size = 100,
#   chunk_overlap = 0
  
# )

# result = splitter.split_text(text)

# print(result)

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('01_book.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
  chunk_size = 200,
  chunk_overlap = 0,
  separator=''
)

result = splitter.split_documents(docs)

print(result[0].page_content)