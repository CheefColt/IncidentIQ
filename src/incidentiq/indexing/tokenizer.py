import re


# Characters considered part of a searchable log token.
#
# Examples:
#   instruction
#   cache
#   fpr29
#   0xffffffff
#   172.16.96.116:33569
TOKEN_PATTERN = re.compile(
    r"[a-z0-9_./:-]+"
)


def tokenize(text: str) -> list[str]:
    """
    Convert a log message into normalized searchable terms.

    The tokenizer:
    - converts text to lowercase
    - extracts alphanumeric and log-relevant symbols
    - preserves values such as IP addresses, ports, paths, and hex values

    Examples
    --------
    "instruction cache parity error"
        -> ["instruction", "cache", "parity", "error"]

    "fpr29=0xffffffff"
        -> ["fpr29", "0xffffffff"]

    "172.16.96.116:33569"
        -> ["172.16.96.116:33569"]
    """

    return TOKEN_PATTERN.findall(
        text.lower()
    )