import pyterrier as pt
from tqdm import tqdm
import json
import os
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])


def rank(index, queries, result):
    index = pt.IndexFactory.of(index)
    queries = queries = pt.io.read_topics(queries)

    bm25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True)
    rm3_pipe = bm25 >> pt.rewrite.RM3(index) >> bm25

    ranking = rm3_pipe(queries)

    pt.io.write_results(res=ranking, filename=result)