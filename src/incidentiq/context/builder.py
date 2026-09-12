from collections import defaultdict
from incidentiq.context.signals import (
    extract_statistics,
    extract_time_range,
    cluster_events_by_time
)

class ContextBuilder:

    def __init__(self, df):

        self.df = df


    def build(
        self,
        results,
        max_evidence=10
    ):
        """
        Convert retrieved search results into structures incident evidence
        """

        evidence = []

        for result in results[:max_evidence]:
            doc_id = result["doc_id"]

            row = self.df.loc[doc_id]

            evidence.append({
                "doc_id" : doc_id,
                "timestamp" : row["timestamp"],
                "node" : row["node"],
                "severity" : row["severity"],
                "message" : row["message"]
            })

        return evidence

    def group_by_message(
            self,
            evidence
    ):
        """
        Group evidence items that have the same log message.
        """

        groups = defaultdict(list)

        for item in evidence:

            groups[item["message"]].append(
                item
            )

            grouped = []

        for message, items in groups.items():
            grouped.append({
                "message":message,
                "count": len(items),
                "occurrences":items
            })

        return grouped


    def build_timeline(
            self,
            evidence
    ):

        """
        Sort retreive evidence chronologically to construct an incident timeline.
        """

        timeline = sorted(
            evidence,
            key = lambda item : item["timestamp"]
        )


        return timeline

    def build_context(
        self,
        results,
        max_evidence=10
    ):
      """
      Build a complete incident context.
      """
  
      evidence = self.build(
          results,
          max_evidence=max_evidence
      )
  
      groups = self.group_by_message(
          evidence
      )
  
      timeline = self.build_timeline(
          evidence
      )
  
      context = {
          "evidence": evidence,
          "groups": groups,
          "timeline": timeline
      }
  
      context["statistics"] = (
          extract_statistics(context)
      )
  
      context["time_range"] = (
          extract_time_range(context)
      )
  
      context["temporal_clusters"] = (
          cluster_events_by_time(context)
      )
  
      return context
