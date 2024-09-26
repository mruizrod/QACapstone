import os
import nest_asyncio  # noqa: E402
nest_asyncio.apply()
import openai


from IPython.display import Markdown, display

# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the API key
#llamaparse_api_key = "llx-p7DLM1NsIUHGEcqYm9UZcFZaiVvRpU7rh8bbbMMR8wo0INcA"
# set up parser
parser = LlamaParse(
    api_key=llamaparse_api_key,
    result_type="markdown"       # Specify temperature # "markdown" and "text" are available
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['data/am-mid-year-outlook-2024.pdf'], file_extractor=file_extractor).load_data()


# create an index from the parsed markdown
index = VectorStoreIndex.from_documents(documents)
# create a query engine for the index
query_engine = index.as_query_engine(
    openai_model="gpt-4o",  # Specify the OpenAI model
    openai_temperature=0.5,        # Optional: Adjust temperature
    openai_api_key=OPENAI_API_KEY # Pass the OpenAI API key if needed
)

# query the engine
query = "What is the key takeaway of Goldman's mid-year outlook 2024?"
response = query_engine.query(query)
print(response)
display(Markdown(f"<b>{response}</b>"))

