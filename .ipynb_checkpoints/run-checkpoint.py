from sentence_transformers import CrossEncoder
import logging
import os
import pandas as pd
import torch, gc
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from financerag.rerank import CrossEncoderReranker
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder, BM25Retriever, HybridRetriever
from financerag.tasks import ConvFinQA, FinanceBench, FinDER, FinQA, FinQABench, MultiHiertt, TATQA
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--encoding_batch_size', type=int, default=8)
parser.add_argument('--rerank1_batch_size', type=int, default=32)
parser.add_argument('--rerank2_batch_size', type=int, default=8)
args = parser.parse_args()

###################################################################################
convfinqa_task = ConvFinQA()
finbench_task = FinanceBench()
finder_task = FinDER()
finqa_task = FinQA()
finqabench_task = FinQABench()
multih_task = MultiHiertt()
tatqa_task = TATQA()
###################################################################################
# Encoder & retrieval
base_encoder = "BAAI/bge-m3" # "intfloat/multilingual-e5-large-instruct"  #"BAAI/bge-large-en-v1.5" #"nvidia/NV-Embed-v2"(20GB) "intfloat/e5-mistral-7b-instruct"(9GB) 
                                            #"dunzhang/stella_en_1.5B_v5" (6GB)  "jinaai/jina-embeddings-v3"(1.1GB) "jinaai/jina-embeddings-v2-base-code"(320MB)
encoder_model = SentenceTransformerEncoder(
    model_name_or_path=base_encoder,
    query_prompt='query: ',
    doc_prompt='passage: '
)

retrieval_model = DenseRetrieval(
    model=encoder_model,
    batch_size=args.encoding_batch_size
 )
###################################################################################
# Run Retrieval
print("Working on ConvfinQA Task")
convfinqa_result = convfinqa_task.retrieve(
    retriever=retrieval_model)
torch.cuda.empty_cache()
print("Working on FinBench Task")
finbench_result = finbench_task.retrieve(
    retriever=retrieval_model)
torch.cuda.empty_cache()
print("Working on FinDER Task")
finder_result = finder_task.retrieve(
    retriever=retrieval_model)
torch.cuda.empty_cache()
print("Working on FinQA Task")
finqa_result = finqa_task.retrieve(
    retriever=retrieval_model)
torch.cuda.empty_cache()
print("Working on FinQABench Task")
finqabench_result = finqabench_task.retrieve(
    retriever=retrieval_model)
torch.cuda.empty_cache()
print("Working on MultiHiertt Task")
multih_result = multih_task.retrieve(
    retriever=retrieval_model)
torch.cuda.empty_cache()
print("Working on TATQA Task")
tatqa_result = tatqa_task.retrieve(
    retriever=retrieval_model)
torch.cuda.empty_cache()

results = [
    convfinqa_result,
    finbench_result,
    finder_result,
    finqa_result,
    finqabench_result,
    multih_result,
    tatqa_result
]
###################################################################################
def get_evalset(dataset_name):
    qrels = {}
    df_qrels = pd.read_csv(f"./data/test/{dataset_name}_qrels.tsv", sep='\t')
    for _, row in df_qrels.iterrows():
        if row['query_id'] not in qrels:
            qrels[row['query_id']] = {}
        qrels[row['query_id']][row['corpus_id']] = row['score']
    return qrels

tasks = [
    convfinqa_task,
    finbench_task,
    finder_task,
    finqa_task,
    finqabench_task,
    multih_task,
    tatqa_task
]

qrels = [
    get_evalset('ConvFinQA'),
    get_evalset('FinanceBench'),
    get_evalset('FinDER'),
    get_evalset('FinQA'),
    get_evalset('FinQABench'),
    get_evalset('MultiHeirtt'),
    get_evalset('TATQA')
]

ndcg_values = []
map_values = []
recall_values = []
precision_values = []

dataset_names = [
    'ConvFinQA',
    'FinanceBench',
    'FinDER',
    'FinQA',
    'FinQABench',
    'MultiHeirtt',
    'TATQA'
]

for qrel, task in zip(qrels, tasks):
    metrics = task.evaluate(qrels=qrel, results=task.retrieve_results, k_values=[10])
    ndcg_values.append(metrics[0]['NDCG@10'])  # NDCG@10

plt.figure(figsize=(12, 8))
plt.plot(ndcg_values, marker='o', label='NDCG@10', color='b')
plt.title('NDCG@10')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(range(len(dataset_names)), dataset_names, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
###################################################################################
base_reranker = "BAAI/bge-reranker-v2-m3" #"BAAI/bge-reranker-base" #m2 

reranker = CrossEncoderReranker(
    model=CrossEncoder(base_reranker))
###################################################################################
# Step 6: Perform reranking 1
batch_size = args.rerank1_batch_size # 32

print("\nWorking on ConvFinQA Reranking")
top_k = 0.6*len(convfinqa_task.retrieve_results)
convfinqa_rerank = convfinqa_task.rerank(
    reranker=reranker,
    results=convfinqa_result,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on FinBench Reranking")
top_k = 0.8*len(finbench_task.retrieve_results)
finbench_rerank = finbench_task.rerank(
    reranker=reranker,
    results=finbench_result,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on FinDER Reranking")
top_k = 0.4*len(finder_task.retrieve_results)
finder_rerank = finder_task.rerank(
    reranker=reranker,
    results=finder_result,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on FinQA Reranking")
top_k = 0.6*len(finqa_task.retrieve_results)
finqa_rerank = finqa_task.rerank(
    reranker=reranker,
    results=finqa_result,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on FinQABench Reranking")
top_k = 0.8*len(finqabench_task.retrieve_results)
finqabench_rerank = finqabench_task.rerank(
    reranker=reranker,
    results=finqabench_result,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on MultiHiertt Reranking")
top_k = 0.4*len(multih_task.retrieve_results)
multih_rerank = multih_task.rerank(
    reranker=reranker,
    results=multih_result,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on TATQA Reranking")
top_k = 0.6*len(tatqa_task.retrieve_results)
tatqa_rerank = tatqa_task.rerank(
    reranker=reranker,
    results=tatqa_result,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
reranking_results = [
    convfinqa_rerank,
    finbench_rerank,
    finder_rerank,
    finqa_rerank,
    finqabench_rerank,
    multih_rerank,
    tatqa_rerank
]
###################################################################################
retrieval_ndcg_values = []
rerank_ndcg_values = []

retrieval_map_values = []
rerank_map_values = []

retrieval_recall_values = []
rerank_recall_values = []

retrieval_precision_values = []
rerank_precision_values = []

# 데이터셋 이름 리스트
dataset_names = [
    'ConvFinQA',
    'FinanceBench',
    'FinDER',
    'FinQA',
    'FinQABench',
    'MultiHeirtt',
    'TATQA'
]

for qrel, task in zip(qrels, tasks):
    # 평가지표 평가 (retrieval 결과)
    #retrieval_metrics = task.evaluate(qrels=qrel, results=task.retrieve_results, k_values=[10])
    
    # 평가지표 평가 (rerank 결과)
    rerank_metrics = task.evaluate(qrels=qrel, results=task.rerank_results, k_values=[10])

    # retrieval_ndcg_values.append(retrieval_metrics[0]['NDCG@10'])
    # retrieval_map_values.append(retrieval_metrics[1]['MAP@10'])
    # retrieval_recall_values.append(retrieval_metrics[2]['Recall@10'])
    # retrieval_precision_values.append(retrieval_metrics[3]['P@10'])
    
    rerank_ndcg_values.append(rerank_metrics[0]['NDCG@10'])
    rerank_map_values.append(rerank_metrics[1]['MAP@10'])
    rerank_recall_values.append(rerank_metrics[2]['Recall@10'])
    rerank_precision_values.append(rerank_metrics[3]['P@10'])

# 그래프 생성
plt.figure(figsize=(14, 10))
#plt.plot(retrieval_ndcg_values, marker='o', label='Retrieval NDCG@10', color='b')
plt.plot(rerank_ndcg_values, marker='x', label='Rerank NDCG@10', color='b', linestyle='--')
plt.title('NDCG@10 Comparison')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(range(len(dataset_names)), dataset_names, rotation=45)  # X축에 데이터셋 이름을 설정
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
###################################################################################
base_reranker2 = "BAAI/bge-reranker-v2-m3" #"BAAI/bge-reranker-v2-m3" #"jinaai/jina-reranker-v2-base-multilingual" #'cross-encoder/ms-marco-MiniLM-L-12-v2' #"BAAI/bge-reranker-base"

reranker2 = CrossEncoderReranker(
    model=CrossEncoder(base_reranker2))
###################################################################################
# Step 6: Perform reranking

top_k= 20 # Number of Reranking results
batch_size = args.rerank2_batch_size # 32

print("\nWorking on ConvFinQA Reranking")
convfinqa_rerank_second = convfinqa_task.rerank(
    reranker=reranker2,
    results=convfinqa_rerank,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on FinBench Reranking")
finbench_rerank_second = finbench_task.rerank(
    reranker=reranker2,
    results=finbench_rerank,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on FinDER Reranking")
finder_rerank_second = finder_task.rerank(
    reranker=reranker2,
    results=finder_rerank,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on FinQA Reranking")
finqa_rerank_second = finqa_task.rerank(
    reranker=reranker2,
    results=finqa_rerank,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on FinQABench Reranking")
finqabench_rerank_second = finqabench_task.rerank(
    reranker=reranker2,
    results=finqabench_rerank,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on MultiHiertt Reranking")
multih_rerank_second = multih_task.rerank(
    reranker=reranker2,
    results=multih_rerank,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
print("\nWorking on TATQA Reranking")
tatqa_rerank_second = tatqa_task.rerank(
    reranker=reranker2,
    results=tatqa_rerank,
    top_k=top_k,  # Rerank the top 100 documents
    batch_size=batch_size
)
torch.cuda.empty_cache()
reranking_results_second = [
    convfinqa_rerank_second,
    finbench_rerank_second,
    finder_rerank_second,
    finqa_rerank_second,
    finqabench_rerank_second,
    multih_rerank_second,
    tatqa_rerank_second
]
###################################################################################
rerank_second_ndcg_values = []

dataset_names = [
    'ConvFinQA',
    'FinanceBench',
    'FinDER',
    'FinQA',
    'FinQABench',
    'MultiHeirtt',
    'TATQA'
]

for qrel, task in zip(qrels, tasks):
    rerank_second_metrics = task.evaluate(qrels=qrel, results=task.rerank_results, k_values=[10])
    rerank_second_ndcg_values.append(rerank_second_metrics[0]['NDCG@10'])

plt.figure(figsize=(14, 10))
plt.plot(retrieval_ndcg_values, marker='o', label='Retrieval NDCG@10', color='b')
plt.plot(rerank_ndcg_values, marker='x', label='Rerank NDCG@10', color='b', linestyle='--')
plt.plot(rerank_second_ndcg_values, marker='s', label='Rerank Second NDCG@10', color='b', linestyle='-.')
plt.title('NDCG@10 Comparison')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(range(len(dataset_names)), dataset_names, rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
###################################################################################
# Step 7: Save results
results_dir = './financerag/results/'
subfolder = "submission_" + str(args.time)  #매번 바꿔줘야함

output_dir = results_dir+subfolder
convfinqa_task.save_results(output_dir=output_dir)
finbench_task.save_results(output_dir=output_dir)
finder_task.save_results(output_dir=output_dir)
finqa_task.save_results(output_dir=output_dir)
finqabench_task.save_results(output_dir=output_dir)
multih_task.save_results(output_dir=output_dir)
tatqa_task.save_results(output_dir=output_dir)

csv_files = [
    output_dir + '/ConvFinQA/results.csv',
    output_dir + '/FinanceBench/results.csv',
    output_dir + '/FinDER/results.csv',
    output_dir + '/FinQA/results.csv',
    output_dir + '/FinQABench/results.csv',
    output_dir + '/MultiHiertt/results.csv',
    output_dir + '/TAT-QA/results.csv'
]

results_df = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(results_df, ignore_index=False)
combined_df.to_csv(output_dir+'/'+subfolder+'.csv', index=False)