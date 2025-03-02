from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents = [
    "Virat kohli is an indian cricketer known for his aggressive batting and leadership.",
    "Ms Dhoni is a former indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records. ",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries. ",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."    
]

query = 'Tell me about Virat kohli'

document_embeddings = embeddings.embed_documents(documents)
query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity([query_embeddings], document_embeddings)[0]


index , score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)