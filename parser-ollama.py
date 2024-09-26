# bring in our LLAMA_CLOUD_API_KEY
import os
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio  # noqa: E402
nest_asyncio.apply()

# bring in deps
from llama_parse import LlamaParse  # noqa: E402
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# set up parser
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
parser = LlamaParse(
    api_key=llamaparse_api_key,
    result_type="markdown"  # "markdown" and "text" are available
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['data/am-mid-year-outlook-2024.pdf'], file_extractor=file_extractor).load_data()

documents

#len(documents)

#documents[0].text
documents[0].text[:200]
print('d')

########### Ollama Models ###############

# by default llamaindex uses OpenAI models
from llama_index.embeddings.ollama import OllamaEmbedding  # noqa: E402

# embed_model = OllamaEmbedding(
#     #model_name="nomic-embed-text",
#     model_name="mistral",
#     base_url="http://localhost:11434",
#     ollama_additional_kwargs={"mirostat": 0},
# )

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

from llama_index.llms.ollama import Ollama  # noqa: E402
llm = Ollama(model="llama2", request_timeout=30.0)

from llama_index.core import Settings  # noqa: E402

Settings.llm = llm
Settings.embed_model = embed_model

# get the answer out of it
# create an index from the parsed markdown
index = VectorStoreIndex.from_documents(documents)
print('i')
# create a query engine for the index
query_engine = index.as_query_engine()
print('q')
# query the engine
from IPython.display import Markdown, display  # noqa: E402

# query the engine
query = "Can small caps finally outperform?"
response = query_engine.query(query)
display(Markdown(f"<b>{response}</b>"))
