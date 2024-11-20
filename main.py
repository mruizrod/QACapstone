import sys
sys.path.append("./src")
from src.pipeline import Pipeline
import streamlit as st
# import tracemalloc; tracemalloc.start()

parser = "unstructured"
model = "nlp_langchain"

pipeline = Pipeline(
    parser=parser, model=model,
    data_path="./data", verbose=True,
); pipeline.train()


# question = "What is the anticipated rate at end-2024 according to the FED?"
# response, score = pipeline.answer(question)
# print(f"A: {response}\n\nSimilarity Score: {score}")


#! currently not working
# def answer(message, history):
#     response, _ = pipeline.answer(message)
#     return response

# import gradio as gr
# gr.ChatInterface(answer, type="messages").launch(share=True)


# Streamlit Interface
st.title("Chatbot Interface with Pipeline")
# Store chat history
if "history" not in st.session_state:
    st.session_state.history = []
# User input
user_input = st.text_input("You:", placeholder="Ask me a question...")
# Handle button press
if st.button("Submit") and user_input:
    # Process user question
    response, score = pipeline.answer(user_input)
    st.session_state.history.append({"question": user_input, "response": response, "score": score})
    user_input = ""  # Clear the input box after submitting
# Display chat history
if st.session_state.history:
    for chat in st.session_state.history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Chatbot:** {chat['response']}")
        st.caption(f"Similarity Score: {chat['score']}")
