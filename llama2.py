from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_parse import LlamaParse

#! check point 1
logger.info("Start parsing documents...")
documents = SimpleDirectoryReader(
    input_dir="./data/",
    file_extractor={".pdf": LlamaParse(result_type="markdown")},
).load_data()
logger.info("Finish parsing documents...")

#! check point 2
logger.info("Initializing Llamma-index...")
# embed_model = OllamaEmbedding(
#     model_name="llama2",
#     base_url="http://localhost:11434",
#     ollama_additional_kwargs={"mirostat": 0},
# )
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="llama2", request_timeout=360.0)
Settings.llm, Settings.embed_model = llm, embed_model
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

#! check point 3
logger.info("Start generating response...")
query = "What is the key takeaway of Goldman's mid-year outlook 2024?"
response = query_engine.query(query)
print(response)
logger.info("Completed.")