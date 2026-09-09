from incidentiq.indexing.tokenizer import tokenize


def parse_query(query: str) -> dict:
    """
    Parse a search query into normal terms and quoted phrases.

    Normal terms are returned separately from quoted phrases.

    Example
    -------
    cache "parity error"

    becomes:

        {
            "terms": ["cache"],
            "phrases": [["parity", "error"]]
        }
    """

    terms: list[str] = []
    phrases: list[list[str]] = []

    inside_quotes = False

    current_phrase: list[str] = []
    current_text: list[str] = []

    def flush_text() -> None:
        """
        Tokenize the currently buffered text and add it
        either to the normal terms or the current phrase.
        """

        if not current_text:
            return

        text = "".join(
            current_text
        )

        tokens = tokenize(text)

        if inside_quotes:
            current_phrase.extend(
                tokens
            )
        else:
            terms.extend(
                tokens
            )

        current_text.clear()

    for char in query:

        # --------------------------------------------------------------
        # Quoted phrase
        # --------------------------------------------------------------

        if char == '"':

            if inside_quotes:

                # Closing quote:
                # finish and store the phrase.
                flush_text()

                if current_phrase:
                    phrases.append(
                        current_phrase
                    )

                current_phrase = []

            else:

                # Opening quote:
                # flush any normal text before it.
                flush_text()

            inside_quotes = not inside_quotes

        # --------------------------------------------------------------
        # Whitespace
        # --------------------------------------------------------------

        elif char.isspace():

            flush_text()

        # --------------------------------------------------------------
        # Normal character
        # --------------------------------------------------------------

        else:

            current_text.append(
                char
            )

    # Flush anything remaining after the loop.
    flush_text()

    return {
        "terms": terms,
        "phrases": phrases,
    }


def get_scoring_terms(
    parsed_query: dict,
) -> list[str]:
    """
    Return all terms that participate in lexical scoring.

    This includes:

        - normal query terms
        - terms contained inside quoted phrases

    Example
    -------
    {
        "terms": ["cache"],
        "phrases": [["parity", "error"]]
    }

    becomes:

        ["cache", "parity", "error"]
    """

    scoring_terms = list(
        parsed_query.get(
            "terms",
            []
        )
    )

    for phrase in parsed_query.get(
        "phrases",
        []
    ):

        scoring_terms.extend(
            phrase
        )

    return scoring_terms


def get_term_candidates(
    terms: list[str],
    positional_index: dict,
) -> set[int]:
    """
    Return documents containing at least one
    supplied term.

    This performs a UNION across the terms.

    Example
    -------
    cache  -> {1, 2, 5}
    error  -> {2, 3, 7}

    result -> {1, 2, 3, 5, 7}
    """

    candidate_docs: set[int] = set()

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
    positional_index: dict,
) -> set[int]:
    """
    Return lexical candidate documents for
    a parsed query.
    """

    scoring_terms = get_scoring_terms(
        parsed_query
    )

    return get_term_candidates(
        scoring_terms,
        positional_index
    )