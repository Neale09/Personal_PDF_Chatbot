from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LocalRetriever:
    """Lightweight, CPU-friendly TF-IDF retrieval; all work stays local."""

    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("No text chunks were provided.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        # Stable ordering for reproducibility.
        indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        results = []
        for index in indices[:top_k]:
            item = self.chunks[index]
            results.append({
                "page": item["page"],
                "text": item["text"],
                "score": float(scores[index]),
            })
        return results
