from loguru import logger
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()
import numpy as np

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter


class NLP_langchain(object):
    def __init__(
        self, 
        chunk_size=200,  #TODO: tune
        chunk_overlap=20,  #TODO: tune
        input_dir=None, 
        input_files=None, 
        verbose=True
    ):
        self.verbose = verbose

        if self.verbose: logger.info("Parsing documents...")
        self.documents = SimpleDirectoryReader(
            input_dir=input_dir, input_files=input_files,
            file_extractor={".pdf": LlamaParse(result_type="markdown")},
        ).load_data()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        document_texts = [doc.text for doc in self.documents]
        self.documents = text_splitter.create_documents(document_texts)

    def train(self):
        if self.verbose: logger.info("Performing embedding...")
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.document_embeddings = [self.embed_model._embed(doc.page_content) for doc in self.documents]

    def answer(self, query, n_context=2):
        query_embedding = self.embed_model._embed(query)
        similarities = cosine_similarity([query_embedding], self.document_embeddings).flatten()
        best_idx = np.argmax(similarities)
        selected_docs = []
        for i in range(n_context+1):
            idx = best_idx + i
            if idx < len(self.documents):
                selected_docs.append(self.documents[idx].page_content)
        response, score = "\n\n".join(selected_docs), similarities[best_idx]
        return response, score
    

if __name__ == "__main__":
    nlp_langchain = NLP_langchain(input_dir="../data/pdf")
    nlp_langchain.train()

    question = "What is the key takeaway of Goldman's mid-year outlook 2024?"
    response, score = nlp_langchain.answer(question)
    print(f"A: {response}\n\nSimilarity Score: {score}")
