import pyterrier as pt
from tqdm import tqdm
import json
import os
if not pt.started():
    pt.init()


def rank(index, queries, result):
    index = pt.IndexFactory.of(index)
    queries = queries = pt.io.read_topics(queries)

    pl2 = pt.BatchRetrieve(index, wmodel="PL2", verbose=True)
    ranking = pl2(queries)

    pt.io.write_results(res=ranking, filename=result)