import os
from loguru import logger
import nest_asyncio;

nest_asyncio.apply()
from dotenv import load_dotenv;

load_dotenv()

from llama_parse import LlamaParse
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from unstructured.chunking.title import chunk_by_title
from langchain_community.document_loaders import UnstructuredPDFLoader

class PDFParser(object):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
    
    def extract_unstructured(self, file_path):
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(file_path, strategy='hi_res')
        #chunks = chunk_by_title(elements)
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
    
    def extract_unstructuredLangchain(self,file_path):
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


if __name__ == '__main__':
    input_dir = "./data/pdf/am-mid-year-outlook-2024.pdf"
    parser = PDFParser()
    data = parser.load_data(input_dir, method = 'llamaparse')
