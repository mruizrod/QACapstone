from utils import process_data
from langchain.text_splitter import RecursiveCharacterTextSplitter
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


class NLP_langchain(object):
    def __init__(
        self,
        parser=None,
        data_path=None,
        chunk_size=200,  # TODO: to be tuned
        chunk_overlap=20,  # TODO: to be tuned
        verbose=True
    ):
        self.verbose = verbose
        if self.verbose:
            logger.info("Loading/Parsing documents...")
        self.documents = process_data(data_path=data_path, method=parser)
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        document_texts = [doc.text for doc in self.documents]
        self.documents = text_splitter.create_documents(document_texts)

    def train(self):
        if self.verbose:
            logger.info("Performing embedding...")
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")
        self.document_embeddings = [self.embed_model._embed(
            doc.page_content) for doc in self.documents]

    def embed(self, query):
        return self.embed_model._embed(query)

    def answer(self, query, n_context=2):
        if self.verbose:
            logger.info("Generating response...")
        query_embedding = self.embed_model._embed(query)
        similarities = cosine_similarity(
            [query_embedding], self.document_embeddings).flatten()
        best_idx = np.argmax(similarities)
        selected_docs = []
        for i in range(n_context+1):
            idx = best_idx + i
            if idx < len(self.documents):
                selected_docs.append(self.documents[idx].page_content)
        response, score = "\n\n".join(selected_docs), similarities[best_idx]
        return response, score
    