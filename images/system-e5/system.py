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

    return index, index_ids


def load_queries(queries_dir):
    files = os.listdir(queries_dir)
    files.sort()

    queries = []
    for file in files:
        if file.endswith(".pt"):
            queries.append(torch.load(queries_dir + "/" + file).numpy())

    with open(queries_dir+"/ids.json", "r") as file:
        querie_ids = json.load(file)

    return np.concat(queries), querie_ids


def write_run(run_tag, topics, I, D, querie_ids, index_ids):
    with open("results/trec/"+run_tag, "w") as f:
        for qid, query, results in zip(topics["qid"].to_list(), I, D):
            for rank, (doc_id, distance) in enumerate(zip(query, results)):
                docno = index_ids[str(doc_id)]
                f.write(f"{qid} Q0 {docno} {rank} {100-distance}".format())



def rank(index, queries, result, batch_size):

    index, index_ids = load_index(index)

    queries


    D, I = index.search(query_embedding.numpy(), k = 1000)

    write_run(run_tag, topics, I, D, index_ids)

    index = pt.IndexFactory.of(index)
    print(queries)
    queries = pt.io.read_topics(queries)


    bm25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True)

    monoT5 = MonoT5ReRanker(verbose=True, batch_size=batch_size)

    pipeline = bm25 >> pt.text.get_text(index, "text", by_query=True) >> monoT5
    ranking = pipeline(queries)

    pt.io.write_results(res=ranking, filename=result)