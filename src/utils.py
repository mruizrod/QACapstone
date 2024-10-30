import os
import pickle
import nest_asyncio; nest_asyncio.apply()
from dotenv import load_dotenv; load_dotenv()

import pdfplumber
from llama_parse import LlamaParse
from llama_index.core import Document
from unstructured.chunking.title import chunk_by_title
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader


def process_data(data_path, method):
    data_dict = load_data(data_path=data_path, method=method)
    split_documents = []
    if method == 'unstructured':
        for docs in data_dict.values():
            for doc in docs:
                metadata = doc.metadata.to_dict()
                split_documents.append(Document(text=doc.text, metadata=metadata))
    if method == 'pdfplumber':
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for docs in data_dict.values():
            split_texts = splitter.split_text(docs)
            for split_text in split_texts:
                split_documents.append(Document(text=split_text))
    if method == 'pypdfloader':
        for docs in data_dict.values():
            for doc in docs:
                metadata = doc.metadata
                split_documents.append(Document(text=doc.page_content,metadata=metadata))
    if method == 'llamaparse':
        for docs in data_dict.values():
            for doc in docs:
                split_documents.append(doc)
    return split_documents


def load_data(data_path, method):
    parsed_data = {}
    data_file = data_path + f"/pkl/{method}.pkl"
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    pdf_path = data_path + "/pdf"
    pdfs = [file for file in os.listdir(pdf_path) if file.endswith('pdf')]
    parser = PDFParser()
    for pdf_name in pdfs:
        if pdf_name in parsed_data: 
            continue
        else:
            print(f"Parsing new document: {pdf_name}")
            full_path = os.path.join(pdf_path, pdf_name)
            document = parser.load_data(full_path, method)
            parsed_data[pdf_name] = document
    with open(data_file, 'wb') as f:
        pickle.dump(parsed_data, f)  
    return parsed_data


class PDFParser(object):
    '''
    PDF parser for single pdf file
    '''
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")

    def extract_unstructured(self, file_path):
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(file_path, strategy='hi_res')
        # chunks = chunk_by_title(elements)
        return elements

    def extract_pdfplumber(self, file_path):
        with pdfplumber.open(file_path) as pdf:
            extracted_data = ""
            for page in pdf.pages:
                extracted_data += page.extract_text() + "\n\n"
        return extracted_data

    def extract_pypdfloader(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def extract_llamaparse(self, file_path):
        parser = LlamaParse(api_key=self.api_key, result_type="markdown")
        return parser.load_data(file_path)

    def extract_unstructuredLangchain(self, file_path):
        loader = UnstructuredPDFLoader(file_path)
        elements = loader.load()
        return elements

    def load_data(self, file_path, method):
        if method == 'pdfplumber':
            return self.extract_pdfplumber(file_path)
        elif method == 'pypdfloader':
            return self.extract_pypdfloader(file_path)
        elif method == 'llamaparse':
            return self.extract_llamaparse(file_path)
        elif method == 'unstructured':
            return self.extract_unstructured(file_path)
        elif method == 'unstructuredLangchain':
            return self.extract_unstructuredLangchain(file_path)
        else:
            raise ValueError(f"Unknown parsing method: {method}")
