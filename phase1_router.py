import numpy as np
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorRouter:
    def __init__(self):
        self.embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.bot_personas = {
            "Bot A (Tech Maximalist)": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
            "Bot B (Doomer / Skeptic)": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
            "Bot C (Finance Bro)": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI.",
        }

        docs = [
            Document(page_content=text, metadata={"bot_id": name})
            for name, text in self.bot_personas.items()
        ]

        self.vector_store = InMemoryVectorStore.from_documents(
            docs, self.embeddings_model
        )

        self.cached_vectors = {
            doc.metadata["bot_id"]: self.embeddings_model.embed_query(doc.page_content)
            for doc in docs
        }

    def _cosine_similarity(self, vec1, vec2):
        """Helper to explicitly compute exact Cosine Similarity as requested"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def route_post_to_bots(self, post_content: str, threshold: float = 0.35):
        """
        Embeds the post and returns only the bots whose persona vector
        matches the post vector with a cosine similarity > threshold.
        """

        post_vector = self.embeddings_model.embed_query(post_content)

        retrieved_docs = self.vector_store.similarity_search(post_content, k=3)

        matched_bots = []
        for doc in retrieved_docs:
            bot_id = doc.metadata["bot_id"]
            bot_vec = self.cached_vectors[bot_id]

            sim = self._cosine_similarity(post_vector, bot_vec)

            if sim > threshold:
                matched_bots.append(
                    {
                        "bot_id": bot_id,
                        "similarity": round(float(sim), 4),
                        "persona": doc.page_content,
                    }
                )

        return sorted(matched_bots, key=lambda x: x["similarity"], reverse=True)


if __name__ == "__main__":
    router = VectorRouter()
    post = "OpenAI just released a new model that might replace junior developers."
    print(f"Routing post: '{post}'")
    matches = router.route_post_to_bots(post)
    for m in matches:
        print(f"- {m['bot_id']} (Sim: {m['similarity']})")
