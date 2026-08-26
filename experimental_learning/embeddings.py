from sentence_transformers import SentenceTransformer, util
import pandas as pd

DATA_PATH = r"E:\incidentiq\data\processed\logs.parquet"

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# texts = [
#     "instruction cache parity error corrected",
#     "cache problem",
#     "disk space is running low",
#     "network connection failed"
# ]

# embeddings = model.encode(texts)

# print(type(embeddings))

df = pd.read_parquet(DATA_PATH)

log_embeddings = model.encode(
    df["message"].tolist(),
    show_progress_bar=True
)

print(log_embeddings.shape)

# similarities = util.cos_sim(
#     embeddings,
#     embeddings
# )

# print(similarities)
query = "cache problem"
query_embedding = model.encode(query, convert_to_tensor=True)

scores = util.cos_sim(
    query_embedding,
    log_embeddings
)

top_k = 10
top_scores, top_indices = scores.topk(top_k)

for score, doc_id in zip(
    top_scores[0],
    top_indices[0]
):
    row = df.iloc[int(doc_id)]

    print(
        f"{score.item():.3f} | "
        f"log_id={row['log_id']} | "
        f"{row['message']}"
    )