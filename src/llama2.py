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
    def __init__(self, parser, data_path, verbose=True):
        """
        Llama2 class for answering questions.

        Args:
            parser (str): The name of the PDF parser to use, one of "unstructured", or "pdfplumber", "pdfloader", or "llamaparse".
            data_path (str): The path from the current file's directory to the data directory.
            verbose (bool, optional): If True, print out logging messages. Defaults to True.
        """
        self.verbose = verbose
        if self.verbose:
            logger.info("Loading/Parsing documents...")
        self.documents = process_data(data_path=data_path, method=parser)

    def train(self):
        """
        Train the Llama2 model.

        This method initializes the Llama2 model by setting its embedding model
        and llm. It also creates a VectorStoreIndex from the parsed documents,
        and sets the query engine to the index.
        """
        if self.verbose:
            logger.info("Initializing Llamma-index with Llama2...")
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model="llama2", request_timeout=360.0)
        Settings.llm, Settings.embed_model = self.llm, self.embed_model
        index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = index.as_query_engine()

    def embed(self, query):
        """
        Perform work embedding for the given query.

        Args:
            query (str): The text input to be embedded.

        Returns:
            numpy.ndarray: The vector representation of the query.
        """
        return self.embed_model._embed(query)

    def answer(self, query):
        """
        Generate an answer to the given query.

        Args:
            query (str): The text input to be answered.

        Returns:
            tuple: A tuple of the response and the similarity score of the
            response to the query.
        """
        if self.verbose:
            logger.info("Generating response...")
        response = self.query_engine.query(query)
        score = cosine_similarity(
            [self.embed_model._embed(query)],
            [self.embed_model._embed(str(response))],
        )[0][0]
        return response, score
