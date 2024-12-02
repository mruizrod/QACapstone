from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.chunking.title import chunk_by_title
from llama_index.core import Document
from llama_parse import LlamaParse
import pdfplumber
from dotenv import load_dotenv
import os
import pickle
import nest_asyncio
nest_asyncio.apply()
load_dotenv()


def process_data(data_path, method):
    data_dict = load_data(data_path=data_path, method=method)
    split_documents = []
    if method == 'unstructured':
        for docs in data_dict.values():
            for doc in docs:
                metadata = doc.metadata.to_dict()
                split_documents.append(
                    Document(text=doc.text, metadata=metadata))
    if method == 'pdfplumber':
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        for docs in data_dict.values():
            split_texts = splitter.split_text(docs)
            for split_text in split_texts:
                split_documents.append(Document(text=split_text))
    if method == 'pypdfloader':
        for docs in data_dict.values():
            for doc in docs:
                metadata = doc.metadata
                split_documents.append(
                    Document(text=doc.page_content, metadata=metadata))
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

def guide(x):
    return f"""
        Objective:
        I will provide you with a task or description. Your job is to output a single, well-structured, actionable prompt that effectively guides another LLMto perform the task.

        Guidelines:

            1.	Direct Prompt Creation:
            •	Do not include explanations or steps about how you engineered the prompt—just provide the final prompt.
            2.	Include Relevant Context:
            •	If the task seems to rely on information from a document or dataset, assume the LLM has access to that data.
            •	Make sure the engineered prompt explicitly refers to that document or dataset to guide the real LLM effectively.
            3.	Clear and Specific Instructions:
            •	Ensure the prompt is concise, grammatically correct, and avoids ambiguity.
            •	Specify the desired format, tone, or any constraints.
            4.	Iterative Refinement:
            •	If a single prompt isn't sufficient, suggest breaking the task into subtasks within the same prompt.

        Examples

        Input:

        “Summarize a financial report and identify key trends.”

        Engineered Prompt:

        “Using the financial report provided, summarize the key trends in revenue, expenses, and profit margins over the past quarter. Highlight any significant changes or patterns in the data. Present your response in a professional tone and limit it to 3-5 sentences.”

        Input:

        “Analyze customer reviews for sentiment.”

        Engineered Prompt:

        “Analyze the customer reviews provided and classify the sentiment as positive, negative, or neutral. Provide a one-sentence explanation for each classification, citing specific phrases from the reviews.”

        Input:

        “Generate a Python function to calculate portfolio variance.”

        Engineered Prompt:

        “Write a Python function that calculates the variance of a portfolio given a list of asset weights and their covariance matrix. Include detailed inline comments explaining each step of the function.”

        Final Instructions:
        When I give you a task, respond only with the final engineered prompt. If the task seems to depend on specific documents or datasets, explicitly reference them in the prompt to guide the LLM (the real one) effectively. Do not include any explanations or additional steps—just the prompt.


        <Prompt: {x}>
    """
