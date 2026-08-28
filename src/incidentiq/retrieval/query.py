from itertools import product


def parse_query(query: str) -> dict:

    terms = []
    phrases = []

    inside_quotes = False
    current_phrase = []
    current_text = []

    for char in query:

        if char == '"':

            if inside_quotes:

                if current_text:
                    current_phrase.append(
                        "".join(current_text)
                    )

                if current_phrase:
                    phrases.append(
                        current_phrase
                    )

                current_phrase = []
                current_text = []

                inside_quotes = False

            else:

                if current_text:

                    terms.append(
                        "".join(current_text)
                    )

                    current_text = []

                inside_quotes = True

        else:

            if char != " ":

                current_text.append(char)

            else:

                if current_text:

                    text = "".join(current_text)

                    if inside_quotes:
                        current_phrase.append(text)
                    else:
                        terms.append(text)

                    current_text = []

    if current_text:

        text = "".join(current_text)

        if inside_quotes:
            current_phrase.append(text)
        else:
            terms.append(text)

    return {
        "terms": terms,
        "phrases": phrases
    }


def get_scoring_terms(parsed_query: dict) -> list[str]:

    scoring_terms = parsed_query["terms"].copy()

    for phrase in parsed_query["phrases"]:
        scoring_terms.extend(phrase)

    return scoring_terms