from utils import process_data
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np
from dotenv import load_dotenv
import nest_asyncio
from loguru import logger
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
nest_asyncio.apply()
load_dotenv()


class NLP_base(object):
    def __init__(
        self,
        parser=None,
        data_path=None,
        verbose=True
    ):
        self.verbose = verbose
        if self.verbose:
            logger.info("Loading/Parsing documents...")
        self.documents = process_data(data_path=data_path, method=parser)

    def train(self):
        if self.verbose:
            logger.info("Performing embedding...")
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")
        self.document_embeddings = [self.embed_model._embed(
            doc.text) for doc in self.documents]

    def embed(self, query):
        return self.embed_model._embed(query)

    def answer(self, query):
        if self.verbose:
            logger.info("Generating response...")
        query_embedding = self.embed_model._embed(query)
        similarities = cosine_similarity(
            [query_embedding], self.document_embeddings).flatten()
        best_idx = np.argmax(similarities)
        response, score = self.documents[best_idx].text, similarities[best_idx]
        return response, score
    