# import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"
from loguru import logger
# import nest_asyncio; nest_asyncio.apply()
# import asyncio; asyncio.set_event_loop(asyncio.new_event_loop())
import tracemalloc; tracemalloc.start()
from dotenv import load_dotenv; load_dotenv()

import ingest
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import utils as chromautils

from langchain_ollama import ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
logger.info("Parsing documents...")
documents, files = ingest.load_parse_data(method = 'unstructured')

# This is the same code than process_data in ingest/utils, the only thing that changes is the definition of Document
# before we used Document from llamaIndex, now is Document from langchain
split_documents = []
for file in files:
    elements = documents[file]
    for element in elements:
            metadata = element.metadata.to_dict()
            split_documents.append(Document(page_content=element.text,metadata=metadata))

docs = chromautils.filter_complex_metadata(split_documents)

#! check point 2
logger.info("Initializing LangChain components...")
# Initialize the embedding model
embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# Initialize the vector store with Chroma
vector_store = Chroma.from_documents(docs, embedding=embed_model)
# Configure the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

#! check point 3
logger.info("Generating response...")
# Initialize LangChain's OllamaLLM with the "llama2" model
llm = ChatOllama(model="llama2", temperature=0)
### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# print(conversational_rag_chain.invoke(
#     {"input": "What is the anticipated rate at end-2024 according to the FED"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )["answer"])
# print("\n")
# print(conversational_rag_chain.invoke(
#     {"input": "Can you explain further?"},
#     config={"configurable": {"session_id": "abc123"}},
# )["answer"])

# Interface
import gradio as gr

def predict(message, history):
    return conversational_rag_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": "abc123"}},
    )["answer"]

gr.ChatInterface(predict, type="messages").launch()
