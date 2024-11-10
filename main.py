from preprocess import Preprocess
from connection import MilvusDbConnection
import os
import pandas as pd
from openai import OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)

# os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

# obj = Preprocess("./research_papers/*")

conn =MilvusDbConnection(db_name="rag_project",collection_name="rag_qa")

# docs = obj.data['text']
# docs = docs.to_list()

# vectors = obj.get_embedding(docs)

# # obj.data.to_pickle("data.pkl")
# # data = pd.read_pickle("data.pkl")
# data = [
#     { "title": obj.data['title'][0],"meta":"placeholder", "text": docs[i],"embedding": vectors[i]}
#     for i in range(len(vectors))
# ]

# conn.insert_data(data)

conn.load_collection("rag_qa")

res = conn.get_load_state("rag_qa")

obj = Preprocess()
# conn =MilvusDbConnection(db_name="rag_project",collection_name="rag_qa")

prompt = ["What are Deep Neural Networks?"]
data = obj.get_embedding(prompt)
res = conn.search(data)

context = []
for i in range(len(res)):
    context.append(res[i]['entity']['text'])

source = res[0]['entity']['title']

def get_chat_completion(context, question):
    # Format the prompt using the provided template
    prompt_template = """
    Answer the following question strictly based on the context provided. ALso list source after the answer.
    
    Here is an example of what the answer should look like:
    '''
    Answer: (The answer to the question)
    Source: (The source of the answer)
    Thank you.
    '''
    here is the context and the question you need to answer:
    Context: {context}

    Source : {source}
    
    Question: {question}
    
    Answer:
    """
    prompt = prompt_template.format(context=context, question=question, source=source)
    print("Prompt:", prompt)

    # Call OpenAI's Chat Completion API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    # Extract and return the answer from the response
    answer = response.choices[0].message.content
    return answer

# Example usage

response = get_chat_completion(context, prompt[0])

print( response)
