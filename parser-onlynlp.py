from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import pdfplumber

# Load environment variables
load_dotenv()

# Function to read and extract text from PDFs
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    paragraphs = text.split(".  \n")
    return paragraphs

def read_pdf_with_plumber(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Checkpoint 1: Document Parsing
logger.info("Start parsing documents...")
document_dir = "./data/"
documents = []

for filename in os.listdir(document_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(document_dir, filename)
        doc_text = read_pdf_with_plumber(file_path)
        documents.append(doc_text)

logger.info("Finish parsing documents.")

# Checkpoint 2: Embedding Documents using TF-IDF
logger.info("Initializing TF-IDF embedding...")
vectorizer = TfidfVectorizer(stop_words='english')
doc_embeddings = vectorizer.fit_transform(documents)
logger.info("Documents embedded.")

# Function to query documents using cosine similarity
def query_documents(query, doc_embeddings, docs):
    # Embed the query using the same TF-IDF vectorizer
    query_embedding = vectorizer.transform([query])

    # Compute cosine similarity between the query and document embeddings
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()

    # Get the document with the highest similarity score
    best_idx = np.argmax(similarities)

    return docs[best_idx], similarities[best_idx]

# Checkpoint 3: Querying the documents
logger.info("Start generating response...")
query = "What is the key takeaway of Goldman's mid-year outlook 2024?"
response, score = query_documents(query, doc_embeddings, documents)
print(f"Response: {response}\nScore: {score}")
logger.info("Completed.")
