import pyterrier as pt
from tqdm import tqdm
import json
import os
if not pt.started():
    pt.init()


def rank(index, queries, result):
    index = pt.IndexFactory.of(index)
    queries = queries = pt.io.read_topics(queries)

    tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF", verbose=True)
    ranking = tfidf(queries)

    pt.io.write_results(res=ranking, filename=result)