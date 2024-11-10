from preprocess import Preprocess
from connection import MilvusDbConnection
import os
import pandas as pd

# os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

obj = Preprocess("./research_papers/*")

conn =MilvusDbConnection(db_name="rag_project",collection_name="rag_qa")

docs = obj.data['text']
docs = docs.to_list()

vectors = obj.get_embedding(docs)

# obj.data.to_pickle("data.pkl")
# data = pd.read_pickle("data.pkl")
data = [
    { "title": obj.data['title'][0],"meta":"placeholder", "text": docs[i],"embedding": vectors[i]}
    for i in range(len(vectors))
]

conn.insert_data(data)

# obj = Preprocess()
# conn =MilvusDbConnection(db_name="rag_project",collection_name="rag_qa")

# prompt = "What is the introduction of the paper?"
# data = obj.get_embedding(prompt)
# res = conn.search(data)

# print(res)
