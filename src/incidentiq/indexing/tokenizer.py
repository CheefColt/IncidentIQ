import re

def tokenize(text: str) -> list[str]:
    """
    Convert a log message into normalized searchable terms.

    Examples:

        "instruction cache parity error"
        -> ["instruction", "cache", "parity", "error"]

        "fpr29=0xffffffff"
        -> ["fpr29", "0xffffffff"]

        "172.16.96.116:33569"
        -> ["172.16.96.116:33569"]
    """

    return re.findall(
        r"[a-z0-9_./:-]+",
        text.lower()
    )