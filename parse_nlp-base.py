from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(query, doc_embeddings, docs):
    query_embedding = embed_model._embed(query)
    similarities = cosine_similarity([query_embedding], doc_embeddings).flatten()
    best_idx = np.argmax(similarities)
    return docs[best_idx].text, similarities[best_idx]

#! check point 1
logger.info("Parsing documents...")
documents = SimpleDirectoryReader(
    input_dir="./data/pdf",
    file_extractor={".pdf": LlamaParse(result_type="markdown")},
).load_data()

#! check point 2
logger.info("Performing embedding...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
document_embeddings = [embed_model._embed(doc.text) for doc in documents]

#! check point 3
logger.info("Generating response...")
query = "Can small caps outperform?"
response, score = search(query, document_embeddings, documents)
print(f"A: {response}\n\nSimilarity Score: {score}")

logger.info("Task completed.")
