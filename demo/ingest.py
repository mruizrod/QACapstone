
import os
import nest_asyncio;
nest_asyncio.apply()
from dotenv import load_dotenv;
load_dotenv()

from llama_parse import LlamaParse
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

import pickle
from ParserClass import PDFParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_parse_data(method):
    data_file = '../data/pkl/'+method+'.pkl'

    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        parsed_data = {}
    
    folder_path = '../data/pdf'
    document_files = [file for file in os.listdir(folder_path) if file.endswith('pdf')]

    parser = PDFParser()

    for file in document_files:
        if file in parsed_data:
            continue
        else:
            document_path = os.path.join(folder_path, file)
            print(f"Parsing document: {file}")
            document = parser.load_data(document_path, method)

            parsed_data[file] = document

    with open(data_file, 'wb') as f:
        pickle.dump(parsed_data, f)  

    return parsed_data, document_files


def process_data(method):
    if method == 'unstructured':
        documents, files = load_parse_data(method = 'unstructured')
        split_documents = []
        for file in files:
            elements = documents[file]
            for element in elements:
                metadata = element.metadata.to_dict()
                split_documents.append(Document(text=element.text,metadata=metadata))
    if method == 'pdfplumber':
        documents, files = load_parse_data(method = 'pdfplumber')
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = []
        for file in files:
            elements = documents[file]
            split_texts = splitter.split_text(elements)
            for split_text in split_texts:
                split_documents.append(
                    Document(text=split_text) 
                )
    if method == 'pypdfloader':
        documents, files = load_parse_data(method = 'pypdfloader')
        split_documents = []
        for file in files:
            elements = documents[file]
            for element in elements:
                metadata = element.metadata
                split_documents.append(Document(text=element.page_content,metadata=metadata))

    if method == 'llamaparse':
        documents, files = load_parse_data(method = 'llamaparse')
        split_documents = []
        for file in files:
            elements = documents[file]
            for element in elements:
                split_documents.append(element)

    return split_documents




if __name__ == '__main__':
    documents, files = load_parse_data(method = 'llamaparse')
    print(files, documents.keys())