from sentence_transformers import SentenceTransformer

# Load a lightweight pretrained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample text to embed
texts = [
    "Hello, this is a test sentence.",
    "Embedding text to vectors is useful for search."
]

# Create embeddings
embeddings = model.encode(texts)

for i, emb in enumerate(embeddings):
    print(f"Embedding {i}:")
    print(emb[:10], "...")  # print first 10 dimensions for brevity
