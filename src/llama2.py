from utils import process_data
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
import nest_asyncio
from loguru import logger
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
nest_asyncio.apply()
load_dotenv()


class Llama2(object):
    def __init__(
        self,
        parser=None,
        data_path=None,
        verbose=True,
    ):
        self.verbose = verbose
        if self.verbose:
            logger.info("Loading/Parsing documents...")
        self.documents = process_data(data_path=data_path, method=parser)

    def train(self):
        if self.verbose:
            logger.info("Initializing Llamma-index with Llama2...")
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model="llama2", request_timeout=360.0)
        Settings.llm, Settings.embed_model = self.llm, self.embed_model
        index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = index.as_query_engine()

    def embed(self, query):
        return self.embed_model._embed(query)

    def answer(self, query):
        if self.verbose:
            logger.info("Generating response...")
        response = self.query_engine.query(query)
        score = cosine_similarity(
            [self.embed_model._embed(query)],
            [self.embed_model._embed(str(response))],
        )[0][0]
        return response, score
