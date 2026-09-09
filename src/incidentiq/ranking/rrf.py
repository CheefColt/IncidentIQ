from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
) -> dict[int, float]:
    """
    Combine multiple ranked document lists using
    Reciprocal Rank Fusion (RRF).

    Each document receives a score based on its rank
    in every ranking where it appears:

        RRF(d) = Σ 1 / (k + rank)

    Parameters
    ----------
    rankings:
        Multiple ranked lists of document IDs.
        Rank 1 is the highest-ranked document.

    k:
        RRF rank constant. The standard value is 60.

    Returns
    -------
    dict[int, float]
        Mapping from document ID to its RRF score.
    """

    if k < 0:
        raise ValueError(
            "RRF constant k must be non-negative."
        )

    scores: dict[int, float] = {}

    for ranking in rankings:

        for rank, doc_id in enumerate(
            ranking,
            start=1,
        ):

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (k + rank)
            )

    return scores