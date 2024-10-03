import sys
sys.path.append("../")
sys.path.append("../src")
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.llama2 import Llama2
from src.nlp_base import NLP_base
from src.nlp_langchain import NLP_langchain
from src.openai import OpenAI

openai = OpenAI(
    input_files=["../data/pdf/am-mid-year-outlook-2024.pdf"],
    verbose=False); openai.train()

llama2 = Llama2(
    input_files=["../data/pdf/am-mid-year-outlook-2024.pdf"],
    verbose=False); llama2.train()

nlp_base = NLP_base(
    input_files=["../data/pdf/am-mid-year-outlook-2024.pdf"],
    verbose=False); nlp_base.train()

nlp_langchain = NLP_langchain(
    input_files=["../data/pdf/am-mid-year-outlook-2024.pdf"],
    verbose=False); nlp_langchain.train()

questions = pd.read_csv("../data/testQ.csv")

# sim_q_r, sim_a_r = [], []
# iterator = questions[["Question", "Answer"]].to_records(index=False)
# for q, a in tqdm(iterator):
#     scores_q, scores_a = [], []
#     for _ in range(10):
#         response, score = openai.answer(q)
#         scores_q.append(score)
#         scores_a.append(cosine_similarity(
#             [openai.embed_model._get_text_embedding(a)], 
#             [openai.embed_model._get_text_embedding(str(response))],
#         )[0][0])
#     sim_q_r.append(np.mean(scores_q))
#     sim_a_r.append(np.mean(scores_a))
# res_openai = questions[["Question"]].copy()
# res_openai["similarity_response_question"] = sim_q_r
# res_openai["similarity_response_answer"] = sim_a_r
# res_openai["model"] = "openai"
#* to save API usage
res_openai = pd.read_csv("./test_results.csv")[lambda x: x.model == "openai"]

def test_model(model):
    sim_q_r, sim_a_r = [], []
    iterator = questions[["Question", "Answer"]].to_records(index=False)
    for q, a in tqdm(iterator):
        scores_q, scores_a = [], []
        for _ in range(10):
            response, score = model.answer(q)
            scores_q.append(score)
            scores_a.append(cosine_similarity(
                [model.embed_model._embed(a)], 
                [model.embed_model._embed(str(response))],
            )[0][0])
        sim_q_r.append(np.mean(scores_q))
        sim_a_r.append(np.mean(scores_a))
    res = questions[["Question"]].copy()
    res["similarity_response_question"] = sim_q_r
    res["similarity_response_answer"] = sim_a_r
    return res

res_llama2 = test_model(llama2)
res_llama2["model"] = "llama2"

res_nlp_base = test_model(nlp_base)
res_nlp_base["model"] = "nlp_base"

res_nlp_langchain = test_model(nlp_langchain)
res_nlp_langchain["model"] = "nlp_langchain"

res = pd.concat([res_openai, res_llama2, res_nlp_base, res_nlp_langchain])
res.sort_values(["Question", "model"], inplace=True, ignore_index=True)
res.to_csv("./test_results.csv", index=False)
