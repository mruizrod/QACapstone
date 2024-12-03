from llama2 import Llama2
from open_ai import OpenAI
from nlp_base import NLP_base
from nlp_langchain import NLP_langchain


def Pipeline(parser, data_path, model, verbose=True):
    """
    Factory function to create a pipeline for answering questions.

    Args:
        parser (str): The name of the PDF parser to use, one of "unstructured", or "pdfplumber", "pdfloader", or "llamaparse".
        data_path (str): The path from the current file's directory to the data directory.
        model (str): The name of the model to use, one of "llama2", "openai", "nlp_base", or "nlp_langchain".
        verbose (bool, optional): If True, print out logging messages. Defaults to True.

    Returns:
        An instance of the requested model class.
    """
    if model == "llama2":
        return Llama2(parser, data_path, verbose=verbose)
    if model == "openai":
        return OpenAI(parser, data_path, verbose=verbose)
    if model == "nlp_base":
        return NLP_base(parser, data_path, verbose=verbose)
    if model == "nlp_langchain":
        return NLP_langchain(parser, data_path, verbose=verbose)
    