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

from langchain.text_splitter import RecursiveCharacterTextSplitter



#! check point 1
logger.info("Start parsing documents...")
documents = SimpleDirectoryReader(
    input_dir="./data/",
    file_extractor={".pdf": LlamaParse(result_type="markdown")},
).load_data()
logger.info("Finish parsing documents...")

logger.info("Initializing langchain...")

text_splitter = RecursiveCharacterTextSplitter(
    #separators=["\n\n", ".", " "],  # Define sentence-level splitting
    separators=["\n\n","."],
    chunk_size=200,  # Or other size
    chunk_overlap=20
)

document_texts = [doc.text for doc in documents]
docs = text_splitter.create_documents(document_texts)
print(len(docs))
#docs1 = text_splitter.split_text(documents)

#! check point 2
logger.info("Initializing embeddings..")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
document_embeddings = [embed_model._embed(doc.page_content) for doc in docs]
logger.info("Documents embedded.")


def query_documents(query, doc_embeddings, docs, n_context = 1):
    # Embed the query
    query_embedding = embed_model._embed(query)

    # Compute cosine similarity between the query and document embeddings
    similarities = cosine_similarity([query_embedding], doc_embeddings).flatten()

    # Get the document with the highest similarity score
    best_idx = np.argmax(similarities)

    selected_docs = []
    for i in range(n_context + 1):
        idx = best_idx + i
        if idx < len(docs):
            selected_docs.append(docs[idx].page_content)


    return "\n\n".join(selected_docs), similarities[best_idx]


#! check point 3
logger.info("Start generating response...")
query = "Can small caps outperform?"
response, score = query_documents(query, document_embeddings, docs)
print(f"Response: {response}\nScore: {score}")
logger.info("Completed.")