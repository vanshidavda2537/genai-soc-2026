from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleVectorStore:

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.documents = []
        self.embeddings = []

    def add(self, document):

        embedding = self.model.encode(document)

        self.documents.append(document)
        self.embeddings.append(embedding)

    def search(self, query, k=3):

        query_embedding = self.model.encode(query)

        scores = []

        for embedding in self.embeddings:

            similarity = cosine_similarity(
                [query_embedding],
                [embedding]
            )[0][0]

            scores.append(similarity)

        results = list(
            zip(self.documents, scores)
        )

        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return results[:k]


store = SimpleVectorStore()

store.add("Python is a programming language.")
store.add("LLMs are trained on large text corpora.")
store.add("RAG combines retrieval and generation.")
store.add("Embeddings map text to vectors.")
store.add("Groq provides fast LLM inference.")


query = "How do we represent text for search?"

results = store.search(query)

print("\nTop Matches:\n")

for doc, score in results:
    print(f"{score:.4f} -> {doc}")