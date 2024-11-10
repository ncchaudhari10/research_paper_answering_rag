from pymilvus import MilvusClient
from pymilvus import model
from preprocess import Preprocess
import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = MilvusClient("milvus_demo.db")

if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=1536,  # The vectors we will use in this demo has 768 dimensions
)

openai_ef = model.dense.OpenAIEmbeddingFunction(
    model_name='text-embedding-3-small', # Specify the model name
    api_key=OPENAI_API_KEY, # Provide your OpenAI API key
    dimensions=1536 # Set the embedding dimensionality
)

obj = Preprocess("./research_papers/*")

docs = obj.data['text']
docs = docs.to_list()

vectors = openai_ef.encode_documents(docs)
print("Dim:", openai_ef.dim, vectors[0].shape)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "paper": obj.data['title'][0]}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))

res = client.insert(collection_name="demo_collection", data=data)

# print(res)

query_vectors = openai_ef.encode_queries(["What are Deep Neural Networks?"])

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=3,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)
