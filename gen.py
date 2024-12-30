from sentence_transformers import CrossEncoder
import logging
import os
import pandas as pd
import torch, gc
from typing import List, Dict
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import openai
# from financerag.rerank import CrossEncoderReranker
# from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder, BM25Retriever, HybridRetriever
from financerag.tasks import ConvFinQA, FinanceBench, FinDER, FinQA, FinQABench, MultiHiertt, TATQA
from financerag.generate import OpenAIGenerator, CustomGenerator
logging.basicConfig(level=logging.INFO)

final_rank = pd.read_csv('./data/final_submission.csv', index_col=False)
final_rank

convfinqa_task = ConvFinQA()
finbench_task = FinanceBench()
finder_task = FinDER()
finqa_task = FinQA()
finqabench_task = FinQABench()
multih_task = MultiHiertt()
tatqa_task = TATQA()

convfinqa_results = final_rank.iloc[:4210]
finder_results = final_rank.iloc[4210:6370]
finqa_results = final_rank.iloc[6370:17840]
finqabench_results = final_rank.iloc[17840:18840]
finbench_results = final_rank.iloc[18840:20340]
multih_results = final_rank.iloc[20340:30080]
tatqa_results = final_rank.iloc[30080:]

confinqa_dict = {}
for key in convfinqa_results['query_id'].unique():
    group = convfinqa_results[convfinqa_results['query_id'] == key]
    confinqa_dict[key] = {value: idx for idx, value in enumerate(group['corpus_id'], start=1)}

finbench_dict = {}
for key in finbench_results['query_id'].unique():
    group = finbench_results[finbench_results['query_id'] == key]
    finbench_dict[key] = {value: idx for idx, value in enumerate(group['corpus_id'], start=1)}

finder_dict = {}
for key in finder_results['query_id'].unique():
    group = finder_results[finder_results['query_id'] == key]
    finder_dict[key] = {value: idx for idx, value in enumerate(group['corpus_id'], start=1)}

finqa_dict = {}
for key in finqa_results['query_id'].unique():
    group = finqa_results[finqa_results['query_id'] == key]
    finqa_dict[key] = {value: idx for idx, value in enumerate(group['corpus_id'], start=1)}

finqabench_dict = {}
for key in finqabench_results['query_id'].unique():
    group = finqabench_results[finqabench_results['query_id'] == key]
    finqabench_dict[key] = {value: idx for idx, value in enumerate(group['corpus_id'], start=1)}

multih_dict = {}
for key in multih_results['query_id'].unique():
    group = multih_results[multih_results['query_id'] == key]
    multih_dict[key] = {value: idx for idx, value in enumerate(group['corpus_id'], start=1)}

tatqa_dict = {}
for key in tatqa_results['query_id'].unique():
    group = tatqa_results[tatqa_results['query_id'] == key]
    tatqa_dict[key] = {value: idx for idx, value in enumerate(group['corpus_id'], start=1)}
    
convfinqa_task.rerank_results = confinqa_dict
finbench_task.rerank_results = finbench_dict
finder_task.rerank_results = finder_dict
finqa_task.rerank_results = finqa_dict
finqabench_task.rerank_results = finqabench_dict
multih_task.rerank_results = multih_dict
tatqa_task.rerank_results = tatqa_dict

########################## Generate Examples ##############################
# print("Start Generation")

# generator = CustomGenerator("meta-llama/Llama-3.2-1B-Instruct")

# example_messages = {
#     "q1": [{"role": "user", "content": "What is AI?"}],
#     "q2": [{"role": "user", "content": "Tell me about Llama model"}]
# }

# answers = generator.generation(example_messages, max_tokens=128, temperature=0.5, top_p=0.7)

# print("===== 결과 =====")
# for q_id, ans in answers.items():
#     print(f"- Query ID: {q_id}\n  Generated Answer: {ans}\n")

########################## Generate Examples ##############################
print("Start Generation")

def custom_messages(
    query: str, 
    documents: List[str]) -> List[Dict]:

    joined_documents = "\n\n".join(
        [f"[Document {i+1}]\n{doc}" for i, doc in enumerate(documents)]
    ) # 이렇게하니깐 인풋토큰 초과 
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial domain expert who provides accurate, "
                "clear, and compliant information based on the provided documents. "
                "When you answer, rely ONLY on the information within the documents. "
                "If the answer is not in the documents, respond that the information "
                "is not found in the provided documents."
                # 답을 직관적으로
            )
        },
        {
            "role": "user",
            "content": (
                f"Here are the retrieved documents:\n\n{joined_documents}"
                "\n\nUsing only the above documents, please answer the following question:"
                f"\n\nQuestion: {query}"
            )
        }
    ]
  
    return messages

######################################################################################

generator = CustomGenerator("gpt-4o-mini") # "meta-llama/Llama-3.2-1B-Instruct"

# convfinqa_task.generate(model=generator, prepare_messages=custom_messages)
# finbench_task.generate(model=generator, prepare_messages=custom_messages)
# finder_task.generate(model=generator, prepare_messages=custom_messages)
# finqa_task.generate(model=generator, prepare_messages=custom_messages)
# finqabench_task.generate(model=generator, prepare_messages=custom_messages)
multih_task.generate(model=generator, prepare_messages=custom_messages)
# tatqa_task.generate(model=generator, prepare_messages=custom_messages)

tasks = [
    # finqa_task,
    # finbench_task,
    # finder_task,
    # finqabench_task,
    multih_task,
    # tatqa_task
]

results_list = []

for task in tasks:
    task_results = task.generate_results  
    task_df = pd.DataFrame(list(task_results.items()), columns=["Q_ID", "Answer"])  
    task_df['Query'] = task.queries.values()  
    results_list.append(task_df)

final_results_df = pd.concat(results_list, ignore_index=True)
final_results_df.to_csv('./results/multih_test.csv', index=False)