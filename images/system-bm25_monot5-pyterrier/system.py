import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker
from tqdm import tqdm
import json
import os
if not pt.started():
    pt.init()


def rank(index, queries, result, batch_size):
    index = pt.IndexFactory.of(index)
    print(queries)
    queries = pt.io.read_topics(queries)


    bm25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True)

    monoT5 = MonoT5ReRanker(verbose=True, batch_size=batch_size)

    pipeline = bm25 >> pt.text.get_text(index, "text", by_query=True) >> monoT5
    ranking = pipeline(queries)

    pt.io.write_results(res=ranking, filename=result)