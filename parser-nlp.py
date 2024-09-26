from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_parse import LlamaParse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#! check point 1
logger.info("Start parsing documents...")
documents = SimpleDirectoryReader(
    input_dir="./data/",
    file_extractor={".pdf": LlamaParse(result_type="markdown")},
).load_data()
logger.info("Finish parsing documents...")

#! check point 2
logger.info("Initializing Llamma-index...")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
document_embeddings = [embed_model._embed(doc.text) for doc in documents]
logger.info("Documents embedded.")


def query_documents(query, doc_embeddings, docs):
    # Embed the query
    query_embedding = embed_model._embed(query)

    # Compute cosine similarity between the query and document embeddings
    similarities = cosine_similarity([query_embedding], doc_embeddings).flatten()

    # Get the document with the highest similarity score
    best_idx = np.argmax(similarities)

    return docs[best_idx].text, similarities[best_idx]


#! check point 3
logger.info("Start generating response...")
query = "Can small caps outperform?"
response, score = query_documents(query, document_embeddings, documents)
print(f"Response: {response}\nScore: {score}")
logger.info("Completed.")