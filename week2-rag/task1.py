from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model=SentenceTransformer('all-MiniLM-L6-v2')

sentence1=input("Enter first senteence: ")
sentence2=input("Enter second sentence: ")

embedding1=model.encode(sentence1)
embedding2=model.encode(sentence2)

similarity=cosine_similarity([embedding1],[embedding2])

print("Cosine Similarity: ",similarity[0][0])
