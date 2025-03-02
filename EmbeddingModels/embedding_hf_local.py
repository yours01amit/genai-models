from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bangol",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

print(str(result))
