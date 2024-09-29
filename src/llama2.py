from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_parse import LlamaParse
from sklearn.metrics.pairwise import cosine_similarity


class Llama2(object):
    def __init__(
        self, 
        input_dir=None, 
        input_files=None, 
        verbose=True,
    ):
        self.verbose = verbose

        if self.verbose: logger.info("Parsing documents...")
        self.documents = SimpleDirectoryReader(
            input_dir=input_dir, input_files=input_files,
            file_extractor={".pdf": LlamaParse(result_type="markdown")},
        ).load_data()

    def train(self):
        if self.verbose: logger.info("Initializing Llamma-index with Llama2...")
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model="llama2", request_timeout=360.0)
        Settings.llm, Settings.embed_model = self.llm, self.embed_model
        index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = index.as_query_engine()

    def answer(self, query):
        if self.verbose: logger.info("Generating response...")
        response = self.query_engine.query(query)
        score = cosine_similarity(
            [self.embed_model._embed(query)], 
            [self.embed_model._embed(str(response))],
        )[0][0]
        return response, score
    

if __name__ == "__main__":
    llama2 = Llama2(input_dir="../data/pdf")
    llama2.train()
    
    question = "What is the key takeaway of Goldman's mid-year outlook 2024?"
    response, score = llama2.answer(question)
    print(f"A: {response}\n\nSimilarity Score: {score}")
