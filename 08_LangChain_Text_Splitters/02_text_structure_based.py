from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Space exploration is the scientific study and discovery of outer space using telescopes, satellites, and spacecraft. It has expanded our understanding of the universe, from studying distant galaxies to exploring planets like Mars and the Moon. 

# Missions conducted by organizations like NASA and ISRO have led to important advancements in technology, communication, and our knowledge of Earth's place in the cosmos. Space exploration also inspires innovation and curiosity, pushing humanity to discover new possibilities beyond our planet.
"""

splitter = RecursiveCharacterTextSplitter(
  chunk_size = 300,
  chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)