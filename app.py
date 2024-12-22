import streamlit as st

from connection import MilvusDbConnection
from preprocess import Preprocess

# Initialize necessary objects
obj = Preprocess()
connection = MilvusDbConnection(db_name="rag_project", collection_name="rag_qa_gi")

# Set streamlit page configuration
st.set_page_config(page_title="Research Papers QA System", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Function to display chat messages
def display_chat():
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div style='text-align: left; padding: 10px; margin: 5px 0; border-radius: 10px; font-family: Arial, sans-serif; font-size: 14px;'>
                    <strong>You:</strong><br>{message['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif message["role"] == "bot":
            st.markdown(
                f"""
                <div style='text-align: left; padding: 10px; margin: 5px 0; border-radius: 10px; font-family: Arial, sans-serif; font-size: 14px;'>
                    <strong>Assistant:</strong><br>{message['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )


# Title and description
st.title("Research Papers QA System")
st.markdown("Ask your questions below and get contextual answers retrieved from the database.")

# Input form for user queries
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input(
        "You:", placeholder="Ask your question here...", label_visibility="hidden"
    )
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    # Add user's message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Generate embedding for the query
    query_embedding = obj.get_embedding(user_input)

    # Search the database for relevant context
    search_results = connection.search(query_embedding)

    # Construct context from search results
    context = "".join(
        [
            f"**Paper**: {data['entity']['title']}\n**Section**: {data['entity']['meta']}\n**Text**: {data['entity']['text']}\n\n"
            for data in search_results
        ]
    )

    # Generate the bot's response based on context and query
    bot_response = obj.get_chat_completion(context, user_input)

    # Add bot's response to chat history
    st.session_state["messages"].append({"role": "bot", "content": bot_response})

# Display chat history
display_chat()

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; font-family: Arial, sans-serif; font-size: 12px;">Powered by RAG and Streamlit</p>
    """,
    unsafe_allow_html=True,
)
