from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [
    "instruction cache parity error corrected",
    "cache problem",
    "disk space is running low",
    "network connection failed"
]

embeddings = model.encode(texts)

# print(type(embeddings))
similarities = util.cos_sim(
    embeddings,
    embeddings
)

print(similarities)