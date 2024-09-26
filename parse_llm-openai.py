import os
from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#! check point 1
logger.info("Parsing documents...")
documents = SimpleDirectoryReader(
    input_dir="./data/pdf",
    file_extractor={".pdf": LlamaParse(result_type="markdown")},
).load_data()

#! check point 2
logger.info("Initializing Llamma-index...")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    openai_model="gpt-4o",
    openai_temperature=0.5,
    openai_api_key=OPENAI_API_KEY,
)

#! check point 3
logger.info("Generating response...")
query = "What is the key takeaway of Goldman's mid-year outlook 2024?"
response = query_engine.query(query)
print(response)

logger.info("Task completed.")
