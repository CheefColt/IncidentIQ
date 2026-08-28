from incidentiq.indexing.tokenizer import tokenize


def parse_query(query: str) -> dict:
    """
    Parse a search query into normal terms and quoted phrases.

    Example:

        cache "parity error"

    becomes:

        {
            "terms": ["cache"],
            "phrases": [["parity", "error"]]
        }
    """

    terms = []
    phrases = []

    inside_quotes = False

    current_phrase = []
    current_text = []

    def flush_text():

        nonlocal current_text

        if not current_text:
            return

        text = "".join(current_text)

        tokens = tokenize(text)

        if inside_quotes:
            current_phrase.extend(tokens)
        else:
            terms.extend(tokens)

        current_text = []

    for char in query:

        # ---------------------------------------------
        # Quote
        # ---------------------------------------------

        if char == '"':

            if inside_quotes:

                # Finish the phrase.
                flush_text()

                if current_phrase:
                    phrases.append(
                        current_phrase
                    )

                current_phrase = []

            else:

                # Start a phrase.
                flush_text()

            inside_quotes = not inside_quotes

        # ---------------------------------------------
        # Whitespace
        # ---------------------------------------------

        elif char.isspace():

            flush_text()

        # ---------------------------------------------
        # Normal character
        # ---------------------------------------------

        else:

            current_text.append(char)

    # Anything remaining after the loop.
    flush_text()

    return {
        "terms": terms,
        "phrases": phrases
    }


def get_scoring_terms(
    parsed_query: dict
) -> list[str]:
    """
    Return all terms that should participate
    in lexical scoring.

    Normal terms + terms inside phrases.

    Example:

        {
            "terms": ["cache"],
            "phrases": [["parity", "error"]]
        }

    becomes:

        ["cache", "parity", "error"]
    """

    scoring_terms = (
        parsed_query["terms"].copy()
    )

    for phrase in parsed_query["phrases"]:

        scoring_terms.extend(
            phrase
        )

    return scoring_terms


def get_term_candidates(
    terms: list[str],
    positional_index: dict
) -> set[int]:
    """
    Return documents containing at least one
    of the supplied terms.

    This is a UNION operation.

    Example:

        cache -> {1, 2, 5}
        error -> {2, 3, 7}

        result -> {1, 2, 3, 5, 7}
    """

    candidate_docs = set()

    for term in terms:

        candidate_docs.update(
            positional_index.get(
                term.lower(),
                {}
            )
        )

    return candidate_docs


def get_candidates(
    parsed_query: dict,
    positional_index: dict
) -> set[int]:
    """
    Get lexical candidate documents for a parsed query.
    """

    scoring_terms = get_scoring_terms(
        parsed_query
    )

    return get_term_candidates(
        scoring_terms,
        positional_index
    )