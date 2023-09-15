import json
import os

import faiss
import pyterrier as pt
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import numpy as np


if not pt.started():
    pt.init()


def load_index(index_dir):
    index = faiss.read_index(index_dir+"/index")
    with open(index_dir+"/ids.json", "r") as file:
        index_ids = json.load(file)

    assert index.ntotal == len(index_ids), "Index: embedding and ids len offsett"
    return index, index_ids


def load_queries(queries_dir):
    files = os.listdir(queries_dir)
    files.sort()

    queries = []
    for file in files:
        if file.endswith(".pt"):
            queries.append(torch.load(queries_dir + "/" + file).numpy())

    with open(queries_dir+"/e5_embeddings-test-ids.json", "r") as file:
        querie_ids = json.load(file)

    queries = np.concatenate(queries)
    assert len(queries) == len(querie_ids), "Queries: embedding and ids len offsett"
    return queries, querie_ids


def write_run(I, D, querie_ids, index_ids, result_path):

    with open(result_path, "w") as f:
        for qid, ranking_doc_ids_faiss, ranking_scores in zip(querie_ids.values(), I, D):
            # for each query
            for rank, (doc_ids_faiss, score) in enumerate(zip(ranking_doc_ids_faiss, ranking_scores)):
                doc_id = index_ids[str(doc_ids_faiss)]
                print(f"{qid} Q0 {doc_id} {rank} {100-score} E5")  # 100-score to map from distance (smaller better) to score, higher better


def rank(index, queries, result_path):

    index, index_ids = load_index(index)
    queries, querie_ids = load_queries(queries)

    D, I = index.search(queries, k = 1000)

    write_run(I, D, querie_ids, index_ids, result_path)