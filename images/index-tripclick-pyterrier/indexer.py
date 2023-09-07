import os

import ir_datasets
import pyterrier as pt
import numpy as np
if not pt.started():
    pt.init()
import json

METHOD = os.getenv("METHOD")

TOPIC_BASE = """<top>
<num>{id}</num>
<title>{title}</title>
</top>\n\n"""


def fix_ir_dataset_naming(dataset):
    return dataset.replace("/", "-")


def move_queries(dataset, index_query_path):
    dataset = ir_datasets.load(dataset)

    with open(f"{index_query_path}/queries.trec", "w") as f:
        for query in dataset.queries_iter():
            f.write(TOPIC_BASE.format(id=query.query_id, title=query.text))
    print("Queries moved")


def load_subcollection_patch_dict():
    with open("tripclick-subcollections.json", "r") as f:
        subcollection_patch_dict = json.load(f)
    return subcollection_patch_dict


def gen_docs(dataset, subcollection):
    subcollection_patch_dict = load_subcollection_patch_dict()

    dataset = ir_datasets.load(dataset)

    for doc in dataset.docs_iter():
        item_subcollection = subcollection_patch_dict.get(doc.doc_id, "000")  # if no metadata, return 0 so it will allways be indexed.
        if isinstance(item_subcollection, float) and np.isnan(item_subcollection):  # some are nan
            item_subcollection = "000"
        if int(item_subcollection[1]) > int(subcollection[1]):
            print(f"Skipping {doc.doc_id}")
            continue

        yield {"docno": doc.doc_id, "text": doc.default_text()}


def index(dataset, index_document_path, index_query_path):
    dataset, subcollection = dataset.split("-")
    move_queries(dataset, index_query_path)

    indexer = pt.index.IterDictIndexer(
        index_path=index_document_path,
        meta={"docno": 26, "text": 100000},
        meta_tags={"text": "ELSE"},
        verbose=True,
    )

    indexref = indexer.index(gen_docs(dataset, subcollection))

    index = pt.IndexFactory.of(indexref)
    print("Indexing done\n_________________________________")
    print(index.getCollectionStatistics().toString())
