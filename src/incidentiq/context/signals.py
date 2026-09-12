from collections import Counter
from datetime import timedelta

def extract_statistics(
        context: dict
) -> dict:

    """
    Extract basic statistics from incident context.
    """

    evidence = context["evidence"]

    if not evidence:
        return{
            "total_events": 0,
            "unqiue_events": 0,
            "affected_nodes" : 0,
            "severity_distribution" : {},
        }

    unique_messages = {
        item["message"]
        for item in evidence
    }

    affected_nodes = {
        item["node"]
        for item in evidence
    }

    severity_distribution = Counter(
        item["severity"]
        for item in evidence
    )

    return {
        "total_events" : len(evidence),
        "unique_messages" : len(unique_messages),
        "affected_nodes" : len(affected_nodes),
        "severity_distribution" : dict(
            severity_distribution
        )
    }

def extract_time_range(
    context : dict
) -> dict :

    """
    Extract the time range covered by the incident evidence
    """

    evidence = context["evidence"]

    if not evidence:
        return {
            "first_event" : None,
            "last_event" : None,
            "duration" : None
        }

    timestamps = [
        item["timestamp"]
        for item in evidence
    ]

    first_event = min(timestamps)
    last_event = max(timestamps)

    duration = { last_event - first_event}

    return {
        "first_event" : first_event,
        "last_event" : last_event,
        "duration" : duration
    }

def cluster_events_by_time(
    context:dict,
    gap_threshold: timedelta = timedelta(hours=24)
) -> list[list[dict]]:

    """
    Group chronological events into temporal clusters.

    Consecutive events belong to the same cluster when their time gap is less than or
    equal to gap_threshold.
    """

    timeline = context["timeline"]

    if not timeline:
        return []

    clusters = []

    current_cluster = [timeline[0]]

    for event in timeline[1:]:

        previous_event = current_cluster[-1]

        gap = (
            event["timestamp"] - previous_event["timestamp"]
        )

        if gap <= gap_threshold:
            current_cluster.append(event)

        else:
            clusters.append(current_cluster)

            current_cluster = [event]

    clusters.append(
        current_cluster
    )

    return clusters