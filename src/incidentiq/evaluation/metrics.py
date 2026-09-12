import math


def precision_at_k(
    results,
    judgments,
    k=10
):
    """
    Calculate Precision@K.
    """

    top_results = results[:k]

    if not top_results:
        return 0.0

    relevant = sum(
        judgments.get(
            result["doc_id"],
            0
        ) > 0
        for result in top_results
    )

    return relevant / len(top_results)


def recall_at_k(
    results,
    judgments,
    k=10
):
    """
    Calculate Recall@K.
    """

    top_results = results[:k]

    relevant_docs = {
        doc_id
        for doc_id, relevance
        in judgments.items()
        if relevance > 0
    }

    if not relevant_docs:
        return 0.0

    retrieved_relevant = sum(
        result["doc_id"] in relevant_docs
        for result in top_results
    )

    return (
        retrieved_relevant
        / len(relevant_docs)
    )


def dcg_at_k(
    results,
    judgments,
    k=10
):
    """
    Calculate Discounted Cumulative Gain.
    """

    dcg = 0.0

    for rank, result in enumerate(
        results[:k],
        start=1
    ):
        relevance = judgments.get(
            result["doc_id"],
            0
        )

        gain = (
            (2 ** relevance) - 1
        )

        discount = math.log2(
            rank + 1
        )

        dcg += gain / discount

    return dcg


def ndcg_at_k(
    results,
    judgments,
    k=10
):
    """
    Calculate Normalized Discounted
    Cumulative Gain at K.
    """

    actual_dcg = dcg_at_k(
        results,
        judgments,
        k
    )

    ideal_relevances = sorted(
        judgments.values(),
        reverse=True
    )[:k]

    ideal_dcg = 0.0

    for rank, relevance in enumerate(
        ideal_relevances,
        start=1
    ):
        gain = (
            (2 ** relevance) - 1
        )

        discount = math.log2(
            rank + 1
        )

        ideal_dcg += (
            gain / discount
        )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg