from sentence_transformers import SentenceTransformer, util


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticRetriever:
    """
    Semantic retriever using Sentence Transformers.

    Document embeddings are generated once when the
    retriever is initialized. Queries are embedded at
    search time and compared using cosine similarity.
    """

    def __init__(
        self,
        df,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.df = df
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

        self.embeddings = self._encode_documents()

    def _encode_documents(self):
        """
        Generate embeddings for all log messages.
        """

        messages = self.df["message"].tolist()

        return self.model.encode(
            messages,
            show_progress_bar=True,
        )

    def _encode_query(
        self,
        query: str,
    ):
        """
        Generate an embedding for a search query.
        """

        return self.model.encode(
            query,
            convert_to_tensor=True,
        )

    def search(
        self,
        query: str,
        top_k: int = 50,
    ):
        """
        Retrieve documents ranked by semantic similarity.

        Parameters
        ----------
        query:
            Natural-language search query.

        top_k:
            Maximum number of documents to return.

        Returns
        -------
        tuple
            (
                ranked document IDs,
                corresponding similarity scores
            )
        """

        query_embedding = self._encode_query(
            query
        )

        scores = util.cos_sim(
            query_embedding,
            self.embeddings,
        )[0]

        result_count = min(
            top_k,
            len(scores),
        )

        top_scores, top_indices = scores.topk(
            result_count
        )

        ranking = [
            int(doc_id)
            for doc_id in top_indices
        ]

        return ranking, top_scores