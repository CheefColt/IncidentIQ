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
        "Analyze the incident using only the supplied evidence.\n"
        "\n"
        "Rules:\n"
        "1. Observations must be directly supported by the supplied evidence.\n"
        "2. Every observation must cite the evidence IDs that support it.\n"
        "3. Hypotheses are allowed to infer possible causes, but must be clearly "
        "labeled as hypotheses and must cite the supporting evidence.\n"
        "4. Do not present an inferred cause as an established fact.\n"
        "5. If evidence for a hypothesis is indirect or weak, assign lower confidence.\n"
        "6. Unknowns should identify important questions that cannot be answered "
        "from the supplied evidence.\n"
        "7. Do not use external knowledge, tools, or assumptions beyond the supplied evidence."
    )

    return "\n".join(lines)