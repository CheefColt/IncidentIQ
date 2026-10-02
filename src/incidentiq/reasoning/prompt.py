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
            f"[{item['doc_id']}] "
            f"{item['timestamp']} | "
            f"{item['node']} | "
            f"{item['severity']} | "
            f"{item['message']}"
        )

    lines.append("")
    lines.append("PATTERNS")

    for pattern in patterns:
        lines.append(str(pattern))

    lines.append("")
    lines.append("INVESTIGATION DECISION")

    lines.append(
        "Determine whether the available evidence is sufficient to "
        "complete the current investigation."
    )

    lines.append(
        "Remaining unknowns do not automatically mean that the "
        "investigation must continue."
    )

    lines.append(
        "For each important unknown, consider whether additional "
        "evidence can realistically be obtained from the available "
        "logs using the search_logs tool."
    )

    lines.append(
        "If an important unknown can be meaningfully investigated "
        "using search_logs, propose a concrete investigation step "
        "and set investigation_complete to false."
    )

    lines.append(
        "If an unknown cannot reasonably be resolved from the "
        "available logs, explicitly preserve it as an unknown and "
        "the investigation may still be marked complete."
    )

    lines.append(
        "Do not continue searching merely because an unknown exists. "
        "Continue only when additional log evidence could meaningfully "
        "reduce the uncertainty."
    )

    lines.append("")
    lines.append("NEXT INVESTIGATION STEPS")

    lines.append(
        "If important investigable unknowns remain, identify concrete "
        "investigation actions that could reduce those unknowns or "
        "distinguish between competing hypotheses."
    )

    lines.append(
        "When an important unknown can be investigated using the "
        "search_logs tool, prefer a specific search query over a "
        "vague instruction."
    )

    lines.append(
        "Do not propose remediation or a conclusion as a next step."
    )

    lines.append("")
    lines.append("TASK")

    lines.append(
        "Analyze the incident using only the supplied evidence and "
        "evidence retrieved through the search_logs tool."
    )

    lines.append(
        "\nRules:\n"
        "1. Observations must be directly supported by the supplied "
        "evidence or tool-retrieved evidence.\n"

        "2. Every observation must cite the evidence IDs that support it.\n"

        "3. Hypotheses may infer possible causes, but must be clearly "
        "labeled as hypotheses and must cite the supporting evidence.\n"

        "4. Do not present an inferred cause as an established fact.\n"

        "5. If evidence for a hypothesis is indirect or weak, assign "
        "lower confidence.\n"

        "6. Unknowns should identify important questions that cannot "
        "currently be answered from the available evidence.\n"

        "7. You may use the search_logs tool when additional log "
        "evidence could meaningfully reduce an important uncertainty. "
        "Do not use external knowledge or assumptions beyond the "
        "supplied evidence and tool results.\n"

        "8. When additional evidence is retrieved using search_logs, "
        "incorporate that evidence into the analysis when relevant. "
        "Tool-retrieved evidence must be cited using its doc_id just "
        "like the initial evidence.\n"

        "9. If an important unknown can still be meaningfully "
        "investigated using search_logs, you MUST propose at least "
        "one concrete next investigation step and set "
        "investigation_complete to false.\n"

        "10. If remaining unknowns cannot reasonably be resolved "
        "using the available logs, do not invent additional searches "
        "just to eliminate those unknowns. The investigation may be "
        "marked complete while explicitly preserving those unknowns.\n"

        "11. A next step must describe an investigation action, not "
        "a conclusion, remediation, or vague instruction. Prefer a "
        "specific search_logs query when additional log evidence "
        "could reduce the uncertainty.\n"

        "12. Each next step must explain why the action is useful and "
        "cite the evidence that motivated it.\n"

        "13. Do not invent a next step merely to satisfy the rules. "
        "If the available evidence is sufficient and there are no "
        "meaningful investigable unknowns, next_steps should be empty "
        "and investigation_complete should be true.\n"

        "14. Do not mark investigation_complete as false merely "
        "because an unknown exists. The remaining unknown must be "
        "something that additional available log evidence could "
        "meaningfully investigate."
    )

    return "\n".join(lines)