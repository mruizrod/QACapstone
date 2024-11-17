import sys
sys.path.append("./src")
from src.pipeline import Pipeline
import tracemalloc; tracemalloc.start()

parser = "unstructured"
model = "nlp_langchain"

pipeline = Pipeline(
    parser=parser, model=model,
    data_path="./data", verbose=True,
); pipeline.train()

question = "What is the anticipated rate at end-2024 according to the FED?"

response, score = pipeline.answer(question)
print(f"A: {response}\n\nSimilarity Score: {score}")

#! currently not working
'''
def answer(message, history):
    response, _ = pipeline.answer(message)
    return response

import gradio as gr
gr.ChatInterface(answer, type="messages").launch(share=True)
'''
