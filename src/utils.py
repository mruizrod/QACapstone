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
    """
    Load parsed data and process it into a list of Document objects.

    Args:
        data_path (str): The path to the directory containing the data.
        method (str): The method to use for processing the data. One of "unstructured", "pdfplumber", "pypdfloader", or "llamaparse".

    Returns:
        A list of Document objects from llama package.
    """
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
    """
    Load parsed data from a given directory, or parse new data and update pickle file if necessary.

    Args:
        data_path (str): The path to the directory containing the data.
        method (str): The method to use for parsing the data. One of "unstructured", "pdfplumber", "pypdfloader", or "llamaparse".

    Returns:
        A dictionary of parsed documents.
    """
    parsed_data = {}
    data_file = data_path + f"/pkl/{method}.pkl"
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    pdf_path = data_path + "/pdf"
    pdfs = [file for file in os.listdir(pdf_path) if file.endswith('pdf') and file not in parsed_data]
    if len(pdfs) > 0:
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
    def __init__(self, api_key=None):
        """
        Initialize a PDFParser object.

        Args:
            api_key (str, optional): The Llama Cloud API key to use. If not provided, the API key will be loaded from the environment variable LLAMA_CLOUD_API_KEY.
        """
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
        """
        Load data from a PDF file using the specified method.

        Args:
            file_path (str): The path to the PDF file to be parsed.
            method (str): The parsing method to use. Must be one of 'pdfplumber', 'pypdfloader', 'llamaparse', 'unstructured', or 'unstructuredLangchain'.

        Returns:
            The extracted data from the PDF file using the specified parsing method.

        Raises:
            ValueError: If the provided method is not recognized.
        """
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

def guide(user_input):
    '''Transform the raw user input to a guided input that contains clearer instructions for the model'''
    return f"""
        Objective:
        I will provide you with a task or description. Your job is to create a single, well-structured, actionable prompt that effectively guides another LLM to perform the task.

        Guidelines:
        1. Direct Prompt Creation:
            • Do not include explanations or steps about how you engineered the prompt—just provide the final prompt.
        2. Include Relevant Context:
            • If the task seems to rely on information from a document or dataset, assume the LLM has access to that data.
            • Make sure the engineered prompt explicitly refers to that document or dataset to guide the real LLM effectively.
        3. Clear and Specific Instructions:
            • Ensure the prompt is concise, grammatically correct, and avoids ambiguity.
            • Specify the desired format, tone, or any constraints.
        4. Iterative Refinement:
            • If a single prompt isn’t sufficient, suggest breaking the task into subtasks within the same prompt.

        When given this prompt, first determine whether it is a general question like "summarize this report" or it is a specific, context-based question like "what is the total increase in EPS this quarter?"

        If it is a general question, then you should give me back a prompt that explicitly give context to the general question. For example, change from "summarize this report" to "summarize this report in a structured way, reference relevant sections and figures in the report."
        If it is a specific question asking for contextual information from the text, then you should give me back a prompt that mentions the fact that they are asking for specific information, and append this at the end: "If you cannot find matching context to this question, say you cannot find matching detail and give the most relevant detail you can find. If you can find matching context to this question, answer the question succinctly and say where you find the information in the report/text"

        Examples

        Input: (general)

        “What is the gist of this report.”

        Engineered Prompt:

        “Summarize the report in a well-structured way and reference details and figures in the report.”

        Input: (general)

        “Analyze customer reviews for sentiment.”

        Engineered Prompt:

        “Analyze the customer reviews provided and classify the sentiment as positive, negative, or neutral.”

        Input: (specific)

        “What is the Fed's predicted issuance of 5-yr bond?”

        Engineered Prompt:

        “Find the Fed's predicted issuance of 5-year bond. Try to find exact matching details. If you cannot find matching context to this question, say you cannot find matching detail and give the most relevant detail you can find. If you can find matching context to this question, answer the question succinctly and say where you find the information in the report/text”

        Final Instructions:
        When I give you a task, respond only with the final engineered prompt. Do not include any explanations or additional steps—just the prompt.


        <Prompt: {user_input}>
    """
