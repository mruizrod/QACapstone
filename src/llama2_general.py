import nest_asyncio
from loguru import logger
from utils import process_data
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
nest_asyncio.apply()
load_dotenv()


class Llama2_General:
    def __init__(self, parser=None, data_path=None, verbose=True):
        """
        Generalized Llama2 class for answering questions. No need for input data.

        Args:
            parser (str, optional): The name of the PDF parser to use. If provided, must be one of "unstructured", or "pdfplumber", "pdfloader", or "llamaparse". Defaults to None.
            data_path (str, optional): The path from the current file's directory to the data directory. Defaults to None.
            verbose (bool, optional): If True, print out logging messages. Defaults to True.
        """
        self.verbose = verbose
        self.memory = []  # Memory to store conversation history
        if data_path and parser:
            if self.verbose:
                logger.info("Loading/Parsing documents...")
            self.documents = process_data(data_path=data_path, method=parser)
        else:
            self.documents = None  # Allow initialization without data
        # Always initialize the embedding and LLM models
        if self.verbose:
            logger.info("Initializing embedding and Llama2 model...")
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model="llama2", request_timeout=360.0)
        self.query_engine = None  # Only set when train() is called

    def train(self):
        """
        Train the generalized Llama2 model, input data is required.

        This method initializes the Llama2 model by setting its embedding model
        and llm. It also creates a VectorStoreIndex from the parsed documents,
        and sets the query engine to the index.
        """
        if not self.documents:
            raise ValueError(
                "No documents to train on. Provide data_path and parser.")
        if self.verbose:
            logger.info("Initializing LlamaIndex with Llama2...")
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
        # Incorporate memory into the prompt
        if self.memory:
            memory_context = "\n".join(
                [f"Q: {q}\nA: {a}" for q, a in self.memory])
            prompt = f"{memory_context}\nQ: {query} A:"
        else:
            prompt = f"Q: {query} A:"
        if self.query_engine:
            # Use query engine if trained on documents
            if self.verbose:
                logger.info(
                    "Generating response using trained query engine...")
            raw_response = self.query_engine.query(prompt)
            response_text = raw_response.response
        else:
            # Use the LLM directly if no query engine is available
            if self.verbose:
                logger.info("Generating response directly via Llama2 model...")
            raw_response = self.llm.complete(prompt=prompt)
            response_text = raw_response.text.strip()
        # Compute similarity score if embedding model is set
        if self.embed_model:
            score = cosine_similarity(
                [self.embed_model._embed(query)],
                [self.embed_model._embed(response_text)],
            )[0][0]
        else:
            score = None
        # Save the current query and response to memory
        self.memory.append((query, response_text))
        return response_text, score
