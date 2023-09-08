import pyterrier as pt
from pyterrier_colbert.ranking import ColBERTFactory
from tqdm import tqdm
import json
import os
if not pt.started():
    pt.init()


def rank(index, queries, result, model_path):
    index = pt.IndexFactory.of(index)
    queries = queries = pt.io.read_topics(queries)

    bm25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True)

    colbert_factory = ColBERTFactory(model_path, None, None)
    colbert = colbert_factory.text_scorer(doc_attr="text")

    pipeline = bm25 >> pt.text.get_text(index, "text", by_query=True) >> colbert
    ranking = pipeline(queries)

    pt.io.write_results(res=ranking, filename=result)