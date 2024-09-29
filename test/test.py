import sys
sys.path.append("../")
sys.path.append("../src")

from src.llama2 import Llama2

llama2 = Llama2(input_dir="../data/pdf")
llama2.train()

question = "What is the key takeaway of Goldman's mid-year outlook 2024?"
response, score = llama2.answer(question)
print(f"A: {response}\n\nSimilarity Score: {score}")
