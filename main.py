import sys
sys.path.append("./src")
from src.pipeline import Pipeline
import streamlit as st





# question = "What is the anticipated rate at end-2024 according to the FED?"
# response, score = pipeline.answer(question)
# print(f"A: {response}\n\nSimilarity Score: {score}")


# Streamlit Interface
st.title("Chatbot Interface with Pipeline")


with st.sidebar:
    st.title('Options')
    selected_reader = st.sidebar.selectbox('Choose the PDF Reader', ['unstructured', 'PDF Loader','PDF Plumber'], key='selected_reader')
    if selected_reader == 'unstructured':
        parser = 'unstructured'
    elif selected_reader == 'PDF Loader':
        parser = 'pypdfloader'
    elif selected_reader == 'PDF PLumber':
        parser = 'pdfplumber'
    st.markdown(
        'We recommend using the reader unstructured for specific questions and PDF Loader for general questions')

    selected_model = st.sidebar.selectbox('Choose the Model', ['LLM', 'Text Extractor'], key='selected_model')
    if selected_model == 'LLM':
        model= 'llama2'

    elif selected_model == 'Text Extractor':
        model = 'nlp_langchain'
    st.markdown(
        'We recommend using the LLM model unless you want to only extract a portion of the text')


pipeline = Pipeline(
    parser=parser, model=model,
    data_path=r'C:\Users\maryj\Documents\Mini5CMU\llamaparser-example\data', verbose=True,
); pipeline.train()

# Store chat history
if "history" not in st.session_state.keys():
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
