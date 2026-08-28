from itertools import product


def phrase_matches(
    phrase: list[str],
    doc_id: int,
    positional_index: dict
) -> bool:

    if not phrase:
        return False

    positions = []

    for term in phrase:

        term_positions = positional_index.get(
            term.lower(),
            {}
        )

        if doc_id not in term_positions:
            return False

        positions.append(
            term_positions[doc_id]
        )

    for combination in product(*positions):

        if all(
            combination[i] + 1 == combination[i + 1]
            for i in range(len(combination) - 1)
        ):
            return True

    return False


def phrase_search(
    phrase: list[str],
    positional_index: dict
) -> set[int]:

    if not phrase:
        return set()

    candidate_docs = None

    for term in phrase:

        docs = set(
            positional_index.get(
                term.lower(),
                {}
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
            positional_index
        )
    }


def minimum_span(
    positions: list[list[int]]
) -> int | None:

    if not positions:
        return None

    best_span = None

    for combination in product(*positions):

        span = max(combination) - min(combination)

        if best_span is None or span < best_span:
            best_span = span

    return best_span


def proximity_score(
    doc_id: int,
    terms: list[str],
    positional_index: dict
) -> float:

    positions = []

    for term in terms:

        term_positions = positional_index.get(
            term.lower(),
            {}
        )

        if doc_id not in term_positions:
            return 0.0

        positions.append(
            term_positions[doc_id]
        )

    span = minimum_span(positions)

    if span is None:
        return 0.0

    if span == 0:
        return 1.0

    return 1.0 / span