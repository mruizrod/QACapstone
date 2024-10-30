import os; os.environ["TOKENIZERS_PARALLELISM"] = "true"
from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()
import os; OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings
from sklearn.metrics.pairwise import cosine_similarity

from utils import process_data


class OpenAI(object):
    def __init__(
        self,
        parser=None,
        data_path=None,
        verbose=True,
    ):
        self.verbose = verbose
        if self.verbose: logger.info("Loading/Parsing documents...")
        self.documents = process_data(data_path=data_path, method=parser)

    def train(self):
        if self.verbose: logger.info("Initializing Llamma-index with OpenAI...")
        self.embed_model = OpenAIEmbedding()
        Settings.embed_model = self.embed_model
        index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = index.as_query_engine(
            openai_model="gpt-4o",
            openai_temperature=0.5,
            openai_api_key=OPENAI_API_KEY,
        )

    def embed(self, query):
        return self.embed_model.get_text_embedding(query)

    def answer(self, query):
        if self.verbose: logger.info("Generating response...")
        response = self.query_engine.query(query)
        score = cosine_similarity(
            [self.embed_model.get_text_embedding(query)], 
            [self.embed_model.get_text_embedding(str(response))],
        )[0][0]
        return response, score
        

if __name__ == "__main__":
    openai = OpenAI(data_path="../data", parser="pypdfloader")
    openai.train()
    
    question = "What is the key takeaway of Goldman's mid-year outlook 2024?"
    response, score = openai.answer(question)
    print(f"A: {response}\n\nSimilarity Score: {score}")
