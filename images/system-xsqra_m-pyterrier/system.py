import pyterrier as pt
from tqdm import tqdm
import json
import os
if not pt.started():
    pt.init()


def rank(index, queries, result):
    index = pt.IndexFactory.of(index)
    queries = queries = pt.io.read_topics(queries)

    XSqrA_M = pt.BatchRetrieve(index, wmodel="XSqrA_M", verbose=True)
    ranking = XSqrA_M(queries)

    pt.io.write_results(res=ranking, filename=result)