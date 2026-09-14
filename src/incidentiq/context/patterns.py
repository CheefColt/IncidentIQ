def repeated_message_patterns(
    context: dict,
    min_occurences: int = 2
) -> list[dict]:

    """
    Identify messages that occur repeatedly within the retreieved evidence.
    """

    patterns = []

    for group in context["groups"]:

        count = group["count"]

        if count < min_occurences:
            continue

        patterns.append({
            "type": "repeated_message",
            "message": group["message"],
            "occurrences": count,
            "evidence_ids": [
                occurrence["doc_id"]
                for occurrence in group["occurrences"]
            ]
        })

    return patterns

def affected_node_patterns(
    context : dict
)-> dict:

    """
    Summarize the number of distinct nodes represented in the evidence.
    """

    statistics = context["statistics"]

    return {
        "type" : "affected_nodes",
        "count" : statistics["affected_nodes"]
    }

def severity_pattern(
    context:dict
)->dict:
    """
    Summarize the severity distribution in the retrieved evidence.
    """

    distribution = (
        context["statistics"]["severity_distribution"]
    )

    return {
        "type" : "severity_distribution",
        "distribution" : distribution
    }


def temporal_patterns(
    context:dict,
    min_cluster_size: int = 2
) -> list[dict]:
    """
    Identify periods where multiple events occured close together in time.
    """

    patterns = []

    for cluster in context["temporal_clusters"]:

        if len(cluster) < min_cluster_size:
            continue

        start = cluster[0]["timestamp"]
        end = cluster[-1]["timestamp"]

        duration = end - start

        patterns.append({
            "type": "temporal_cluster",
            "event_count" : len(cluster),
            "start" : start,
            "end" : end,
            "duration" : duration
        })

    return patterns


def extract_patterns(context:dict) -> list[dict]:
    """
    Extract observable patterns from incident context
    """

    patterns = []

    patterns.extend(
        repeated_message_patterns(context)
    )

    patterns.append(
        affected_node_patterns(context)
    )

    patterns.append(
        severity_pattern(context)
    )

    patterns.extend(
        temporal_patterns(context)
    )

    return patterns