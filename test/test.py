import sys
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(base_dir)  # Parent directory
sys.path.append(os.path.join(base_dir, "src"))  # src directory
data_path = os.path.join(base_dir, "data")


from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline import Pipeline


questions = pd.read_csv(os.path.join(base_dir, "data", "testQ.csv"))

#questions = pd.read_csv("../data/testQ.csv")
models = [ "nlp_base", "nlp_langchain","llama2",]
parsers = ["unstructured","llamaparse", "pdfplumber", "pypdfloader"]

result = pd.DataFrame()
for model in models:
    for parser in parsers:
        pipeline = Pipeline(
            parser=parser, model=model,
            data_path=data_path, verbose=False,
        ); pipeline.train()
        sim_q_r, sim_a_r = [], []
        iterator = questions[["Question", "Answer"]].to_records(index=False)
        for q, a in tqdm(iterator):
            scores_q, scores_a = [], []
            for _ in range(10):
                response, score = pipeline.answer(q)
                scores_q.append(score)
                scores_a.append(cosine_similarity(
                    [pipeline.embed(a)], 
                    [pipeline.embed(str(response))],
                )[0][0])
            sim_q_r.append(np.mean(scores_q))
            sim_a_r.append(np.mean(scores_a))
        res = questions[["Question"]].copy()
        res["model"] = model; res["parser"] = parser
        res["similarity_response_question"] = sim_q_r
        res["similarity_response_answer"] = sim_a_r
        result = pd.concat([result, res])

result.sort_values(["Question", "model", "parser"], inplace=True, ignore_index=True)
result.to_csv(os.path.join(base_dir,"test_results.csv"), index=False)