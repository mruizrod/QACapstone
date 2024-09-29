from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()
import numpy as np

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from sklearn.metrics.pairwise import cosine_similarity


class NLP_base(object):
    def __init__(
        self, 
        input_dir=None, 
        input_files=None, 
        verbose=True
    ):
        self.verbose = verbose

        if self.verbose: logger.info("Parsing documents...")
        self.documents = SimpleDirectoryReader(
            input_dir=input_dir, input_files=input_files,
            file_extractor={".pdf": LlamaParse(result_type="markdown")},
        ).load_data()

    def train(self):
        if self.verbose: logger.info("Performing embedding...")
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.document_embeddings = [self.embed_model._embed(doc.text) for doc in self.documents]

    def answer(self, query):
        if self.verbose: logger.info("Generating response...")
        query_embedding = self.embed_model._embed(query)
        similarities = cosine_similarity([query_embedding], self.document_embeddings).flatten()
        best_idx = np.argmax(similarities)
        response, score = self.documents[best_idx].text, similarities[best_idx]
        return response, score
    

if __name__ == "__main__":
    nlp_base = NLP_base(input_dir="../data/pdf")
    nlp_base.train()

    question = "What is the key takeaway of Goldman's mid-year outlook 2024?"
    response, score = nlp_base.answer(question)
    print(f"A: {response}\n\nSimilarity Score: {score}")
