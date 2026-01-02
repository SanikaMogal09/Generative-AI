from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
# embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Bumrah."

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

# print(cosine_similarity([query_embeddings],doc_embeddings))

scores = cosine_similarity([query_embeddings],doc_embeddings)[0] # 0 so that the result is 1D list

# print(list(enumerate(scores)))

index,score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1] #sorting based on the second parameter, -1 to get the highest score.

print(query)
print(documents[index])
print("Similarity score is :",score)