from llama2 import Llama2
from open_ai import OpenAI
from nlp_base import NLP_base
from nlp_langchain import NLP_langchain


def Pipeline(parser, data_path, model, verbose=True):
    if model == "llama2":
        return Llama2(parser, data_path, verbose=verbose)
    if model == "openai":
        return OpenAI(parser, data_path, verbose=verbose)
    if model == "nlp_base":
        return NLP_base(parser, data_path, verbose=verbose)
    if model == "nlp_langchain":
        return NLP_langchain(parser, data_path, verbose=verbose)
    