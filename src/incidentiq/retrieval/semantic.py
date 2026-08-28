from sentence_transformers import SentenceTransformer, util


class SemanticRetriever:

    def __init__(
        self,
        df,
        model_name: str = "all-MiniLM-L6-v2"
    ):

        self.df = df

        self.model = SentenceTransformer(
            model_name
        )

        self.embeddings = self.model.encode(
            df["message"].tolist(),
            show_progress_bar=True
        )

    def search(
        self,
        query: str,
        top_k: int = 50
    ):

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        scores = util.cos_sim(
            query_embedding,
            self.embeddings
        )[0]

        top_scores, top_indices = scores.topk(
            top_k
        )

        ranking = [
            int(doc_id)
            for doc_id in top_indices
        ]

        return ranking, scores