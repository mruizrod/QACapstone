
import os
import nest_asyncio;
nest_asyncio.apply()
from dotenv import load_dotenv;
load_dotenv()

from llama_parse import LlamaParse
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

import pickle
from ParserClass import PDFParser

# NOTES: Unstructured needs C++ and a specific installation
# NOTES: I am using pickle instead of json given the structure of the documents
# they are not serializable


def load_parse_data(method):
    data_file = './data/parsed_data_'+method+'.pkl'

    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        parsed_data = {}
    
    folder_path = './data/pdf'
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

    return parsed_data


if __name__ == '__main__':
    docs = load_parse_data(method = 'unstructured')
    print(docs)