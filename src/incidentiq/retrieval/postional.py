from itertools import product


def phrase_matches(
    phrase: list[str],
    doc_id: int,
    positional_index: dict,
) -> bool:
    """
    Check whether a phrase occurs exactly and
    consecutively inside a document.

    Parameters
    ----------
    phrase:
        Ordered list of query terms.

    doc_id:
        Document to inspect.

    positional_index:
        Mapping:

            term -> doc_id -> positions

    Returns
    -------
    bool
        True if the phrase occurs exactly in order.
    """

    if not phrase:
        return False

    term_positions = []

    for term in phrase:

        positions = positional_index.get(
            term.lower(),
            {},
        ).get(
            doc_id,
            [],
        )

        if not positions:
            return False

        term_positions.append(
            positions
        )

    for combination in product(
        *term_positions
    ):

        if all(
            left + 1 == right
            for left, right in zip(
                combination,
                combination[1:],
            )
        ):
            return True

    return False


def phrase_search(
    phrase: list[str],
    positional_index: dict,
) -> set[int]:
    """
    Find documents containing an exact phrase.

    Documents must contain every term before
    positional matching is attempted.
    """

    if not phrase:
        return set()

    candidate_docs: set[int] | None = None

    for term in phrase:

        docs = set(
            positional_index.get(
                term.lower(),
                {},
            )
        )

        if candidate_docs is None:
            candidate_docs = docs
        else:
            candidate_docs &= docs

    if not candidate_docs:
        return set()

    return {
        doc_id
        for doc_id in candidate_docs
        if phrase_matches(
            phrase,
            doc_id,
            positional_index,
        )
    }


def minimum_span(
    positions: list[list[int]],
) -> int | None:
    """
    Find the smallest positional span covering
    all query terms.

    The span is:

        max(position) - min(position)

    Example
    -------
    cache  -> [1]
    parity -> [2]
    error  -> [3]

    span = 3 - 1 = 2
    """

    if not positions:
        return None

    best_span: int | None = None

    for combination in product(*positions):

        span = (
            max(combination)
            - min(combination)
        )

        if (
            best_span is None
            or span < best_span
        ):
            best_span = span

    return best_span


def proximity_score(
    doc_id: int,
    terms: list[str],
    positional_index: dict,
) -> float:
    """
    Score the proximity of query terms in a document.

    Smaller positional spans produce larger scores.

        span = 1 -> 1.0
        span = 2 -> 0.5
        span = 4 -> 0.25

    Returns 0.0 when one or more query terms
    are absent from the document.
    """

    if not terms:
        return 0.0

    positions = []

    for term in terms:

        term_positions = positional_index.get(
            term.lower(),
            {},
        )

        doc_positions = term_positions.get(
            doc_id,
            [],
        )

        if not doc_positions:
            return 0.0

        positions.append(
            doc_positions
        )

    span = minimum_span(
        positions
    )

    if span is None:
        return 0.0

    if span == 0:
        return 1.0

    return 1.0 / span