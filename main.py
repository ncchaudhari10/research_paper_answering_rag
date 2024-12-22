from preprocess import Preprocess
from connection import MilvusDbConnection
import pandas as pd


obj = Preprocess("./research_papers/*")

obj.data['embedding'] = obj.data['text'].apply(lambda x: obj.get_embedding(x))

# Apply a row-wise conversion to dictionaries
output = obj.data.apply(lambda row: row.to_dict(), axis=1).tolist()

# print(obj.data.head())
connection = MilvusDbConnection(db_name="rag_project",collection_name="rag_qa_")

connection.insert_data(output)

# obj = Preprocess()
# connection = MilvusDbConnection(db_name="rag_project",collection_name="rag_qa")

# query ="what are deep neural networks?"
# embed = obj.get_embedding(query)

# res = connection.search(embed)


# context = ""
# for data in res:
#     context += f"Paper: {data['entity']['title']}\n"
#     context += f"Section: {data['entity']['meta']}\n"
#     context += f"Text: {data['entity']['text']}\n\n"  # Adding extra line breaks for readability

# # Example usage
# response = obj.get_chat_completion(context, query)

# print(response)