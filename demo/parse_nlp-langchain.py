from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()
import numpy as np

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter


def search(query, doc_embeddings, docs, n_context=2):
    query_embedding = embed_model._embed(query)
    similarities = cosine_similarity([query_embedding], doc_embeddings).flatten()
    best_idx = np.argmax(similarities)
    selected_docs = []
    for i in range(n_context+1):
        idx = best_idx + i
        if idx < len(docs):
            selected_docs.append(docs[idx].page_content)
    return "\n\n".join(selected_docs), similarities[best_idx]


#! check point 1
logger.info(f"Parsing documents...")
documents = SimpleDirectoryReader(
    input_dir="../data/pdf",
    file_extractor={".pdf": LlamaParse(result_type="markdown")},
).load_data()
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n"],
    chunk_size=200,  #TODO: tune
    chunk_overlap=20,
)
document_texts = [doc.text for doc in documents]
docs = text_splitter.create_documents(document_texts)

#! check point 2
logger.info("Performing embedding...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
document_embeddings = [embed_model._embed(doc.page_content) for doc in docs]

#! check point 3
logger.info("Generating response...")
query = "Can small caps outperform?"; print(f"Q: {query}")
response, score = search(query, document_embeddings, docs)
print(f"A: {response}\n\nSimilarity Score: {score}")

logger.info("Task completed.")
