def build_reasoning_prompt(
    query: str,
    context: dict,
    patterns: list[dict]
) -> str:

    lines = []

    lines.append("INCIDENT QUERY")
    lines.append(query)

    lines.append("")
    lines.append("EVIDENCE")

    for item in context["evidence"]:
        lines.append(
            f"[{item["doc_id"]}] "
            f"{item["timestamp"]} | "
            f"{item["node"]} | "
            f"{item["severity"]} | "
            f"{item["message"]} | "
        )


    lines.append("")
    lines.append("PATTERNS")

    for pattern in patterns:
        lines.append(str(pattern))

    lines.append("")
    lines.append("TASK")
    lines.append(
        "Analyze the incident using only the supplied evidence. "
        "Seperate observations from hypotheses and explicitly state "
        "what cannot be determined from the evidence."
    )

    return "\n".join(lines)