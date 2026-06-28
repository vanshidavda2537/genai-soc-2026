from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentencea=input("Sentence A: ")
sentenceb=input("Sentence B: ")

embedding_a=model.encode(sentencea)
embedding_b=model.encode(sentenceb)

similarity=cosine_similarity(
    [embedding_a],
    [embedding_b]
)[0][0]

print(f"\nCosine Similarity: {similarity:.2f}")