import os
import ir_datasets
import pyterrier_doc2query
import pyterrier as pt
if not pt.started():
    pt.init()

METHOD = os.getenv("METHOD")

TOPIC_BASE = """<top>
<num>{id}</num>
<title>{title}</title>
</top>\n\n"""

def fix_ir_dataset_naming(dataset_name):
    return "-".join(dataset_name.split("/")[-2:])


def move_queries(dataset_name, index_query_path):
    dataset = ir_datasets.load(dataset_name)

    with open(f"{index_query_path}/queries.trec", "w") as f:
        for topic in dataset.queries_iter():
            f.write(TOPIC_BASE.format(id=topic.query_id, title=topic.title))


def docs_generator(dataset_name):
    ids = []
    dataset = ir_datasets.load(dataset_name)
    for doc in dataset.docs_iter():
        if doc.doc_id in ids:
            continue
        ids.append(doc.doc_id)
        yield {"docno": doc.doc_id, "text": doc.default_text()}


def index(dataset_name, index_document_path, index_query_path, batch_size, num_samples):
    move_queries(dataset_name, index_query_path)
    
    doc2query = pyterrier_doc2query.Doc2Query(batch_size=batch_size, append=True, num_samples=num_samples, verbose=True, fast_tokenizer=True)

    indexer = pt.index.IterDictIndexer(
        index_path=index_document_path,
        meta={"docno": 26, "text": 100000},
        meta_tags={"text": "ELSE"},
        verbose=True,
        )
    
    pipeline = doc2query >> indexer

    indexref = pipeline.index(docs_generator(dataset_name))
    
    index = pt.IndexFactory.of(indexref)
    print("Indexing done\n_________________________________")
    print(index.getCollectionStatistics().toString())
