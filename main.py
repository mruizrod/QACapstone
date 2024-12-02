import sys
sys.path.append("./src")
from datetime import datetime
from loguru import logger
from src.llama2_general import Llama2_General
from src.pipeline import Pipeline
from src.utils import guide
import streamlit as st


@st.cache_resource
def get_pipeline(parser, model):
    pipeline = Pipeline(
        parser=parser, model=model,
        data_path="./data", verbose=True)
    pipeline.train()
    return pipeline


with st.sidebar:
    st.title("Pipeline Configuration")
    selected_reader = st.sidebar.selectbox(
        "Choose the PDF Reader",
        ["Unstructured", "PDF Loader", "PDF Plumber"],
        key="selected_reader",
    )
    if selected_reader == "Unstructured":
        parser = "unstructured"
    elif selected_reader == "PDF Loader":
        parser = "pypdfloader"
    elif selected_reader == "PDF Plumber":
        parser = "pdfplumber"
    st.markdown(
        "We recommend using the reader unstructured for specific questions and PDF Loader for general questions")

    selected_model = st.sidebar.selectbox(
        "Choose the Model",
        ["LLM", "Text Extractor"],
        key="selected_model",
    )
    if selected_model == "LLM":
        model = "llama2"
    elif selected_model == "Text Extractor":
        model = "nlp_langchain"
    st.markdown(
        "We recommend using the LLM model unless you want to only extract a portion of the text")

pipeline = get_pipeline(parser, model)

st.title("Chatbot for Financial Reports")
if "history" not in st.session_state:
    st.session_state.history = []
user_input = st.text_input("You:", placeholder="Ask me a question...")
if st.button("Submit") and user_input:
    if pipeline.verbose:
        logger.info("Performing prompt engineering...")
    guided_input = guide(user_input)
    engineered_input, _ = Llama2_General(verbose=False).answer(guided_input)
    response, score = pipeline.answer(user_input)
    dt = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "parser": parser, "model": model,
        "question": user_input, "response": response,
        "score": score, "datetime": dt,
    })
    user_input = ""
if st.session_state.history:
    for chat in st.session_state.history[::-1]:
        st.markdown(f"**Question**: {chat['question']}")
        st.markdown(f"**Answer ({chat['parser']}+{chat['model']}):**")
        st.markdown(f"{chat['response']}")
        st.caption(chat["datetime"])
        # st.caption(f"Similarity Score: {chat['score']}")
        st.markdown("---")


# What is the anticipated rate at end-2024 according to the FED?
# What is the key takeaway of Goldman’s mid-year outlook 2024?
