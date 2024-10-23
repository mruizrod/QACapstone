# IDEA
# Implement pickle and load documents (to avoid reading the documents in every item
from loguru import logger
import nest_asyncio;

nest_asyncio.apply()
from dotenv import load_dotenv;

load_dotenv()

from llama_index.core import SimpleDirectoryReader
from unstructured.partition.pdf import partition_pdf
from llama_parse import LlamaParse
import pdfplumber
from langchain.document_loaders import PyPDFLoader


def extract_unstructured(file_path):
    elements = partition_pdf(file_path, strategy='hi_res')
    text = "\n\n".join([str(element) for element in elements])
    print(text)
    return text


def extract_pdfplumber(file_path):
    extracted_data = {
        "text": "",
        "tables": []
    }

    with pdfplumber.open(file_path) as pdf:
        extracted_data = ''
        for page in pdf.pages:
            extracted_data += page.extract_text() + "\n\n"

    return extracted_data


def extract_pypdfloader(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def extract_llamaparse(input_dir):
    file_extractor = {".pdf": LlamaParse(result_type="markdown")}

    # Load documents from the directory
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        file_extractor=file_extractor
    ).load_data()
    return documents


input_dir = r"C:\Users\maryj\Documents\Mini5CMU\llamaparser-example\data\pdf\am-mid-year-outlook-2024.pdf"
input_dir = r"C:\Users\maryj\Documents\Mini5CMU\llamaparser-example\data\pdf"

documents = extract_llamaparse(input_dir)
print(documents)

