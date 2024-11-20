import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from loguru import logger
import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from sklearn.metrics.pairwise import cosine_similarity

from utils import process_data


# class Llama2_General(object):
#     def __init__(
#         self,
#         parser=None,
#         data_path=None,
#         verbose=True,
#     ):
#         self.verbose = verbose
#         if data_path and parser:
#             if self.verbose:
#                 logger.info("Loading/Parsing documents...")
#             self.documents = process_data(data_path=data_path, method=parser)
#         else:
#             self.documents = None  # Allow initialization without data
            
#         # Always initialize the embedding and LLM models
#         if self.verbose:
#             logger.info("Initializing embedding and Llama2 model...")
#         self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#         self.llm = Ollama(model="llama2", request_timeout=360.0)
        
#         self.query_engine = None  # Only set when train() is called

#     def train(self):
#         if not self.documents:
#             raise ValueError("No documents to train on. Provide data_path and parser.")
        
#         if self.verbose:
#             logger.info("Initializing LlamaIndex with Llama2...")
#         Settings.llm, Settings.embed_model = self.llm, self.embed_model
#         index = VectorStoreIndex.from_documents(self.documents)
#         self.query_engine = index.as_query_engine()

#     def embed(self, query):
#         return self.embed_model._embed(query)

#     def answer(self, query):
#         """
#         Generate an answer to a query using the Llama 2 model.

#         Args:
#             query (str): The input query or prompt.

#         Returns:
#             tuple: Response text and similarity score (if embedding model is available).
#         """
#         if self.query_engine:
#             # Use query engine if trained on documents
#             if self.verbose:
#                 logger.info("Generating response using trained query engine...")
#             raw_response = self.query_engine.query(query)
#             response_text = raw_response.response  # Extract only the response text
#         else:
#             # Use the LLM directly if no query engine is available
#             if self.verbose:
#                 logger.info("Generating response directly via Llama2 model...")
#             raw_response = self.llm.complete(prompt=query)
#             response_text = raw_response.text.strip()  # Extract only the response text

#         # Compute similarity score if embedding model is set
#         if self.embed_model:
#             score = cosine_similarity(
#                 [self.embed_model._embed(query)],
#                 [self.embed_model._embed(response_text)],
#             )[0][0]
#         else:
#             score = None

#         return response_text, score

# if __name__ == "__main__":
#     # Example usage without data for prompt engineering
#     llama2 = Llama2_General(verbose=True)  # No data_path or parser needed
#     question = "How do I structure a prompt to summarize financial documents?"
#     response, score = llama2.answer(question)
#     print(f"A: {response}\n\nSimilarity Score: {score}")


class Llama2_General:
    def __init__(
        self,
        parser=None,
        data_path=None,
        verbose=True,
    ):
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
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model="llama2", request_timeout=360.0)
        
        self.query_engine = None  # Only set when train() is called

    def train(self):
        if not self.documents:
            raise ValueError("No documents to train on. Provide data_path and parser.")
        
        if self.verbose:
            logger.info("Initializing LlamaIndex with Llama2...")
        Settings.llm, Settings.embed_model = self.llm, self.embed_model
        index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = index.as_query_engine()

    def embed(self, query):
        return self.embed_model._embed(query)

    def answer(self, query):
        # Incorporate memory into the prompt
        if self.memory:
            memory_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.memory])
            prompt = f"{memory_context}\nQ: {query} A:"
        else:
            prompt = f"Q: {query} A:"

        if self.query_engine:
            # Use query engine if trained on documents
            if self.verbose:
                logger.info("Generating response using trained query engine...")
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