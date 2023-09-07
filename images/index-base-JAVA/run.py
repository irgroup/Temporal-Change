from argparse import ArgumentParser
import os
import ir_datasets
import pyterrier as pt
pt.init()

METHOD = "pyterrier"

def check_artefact_type(artefact_name):
    artefact_type = artefact_name.split("-")[0]
    return artefact_type


def check_artefact_exist(artefact_name):
    artefact_type = check_artefact_type(artefact_name)
    return os.path.exists(f"/IRLab/{artefact_type}/{artefact_name}-{METHOD}")


def setup_index_dir(artefact_name):

    os.makedirs(f"/IRLab/index/index-{artefact_name}-{METHOD}/documents")
    
    os.makedirs(f"/IRLab/index/index-{artefact_name}-{METHOD}/queries")


def fix_ir_dataset_naming(dataset):
    return "-".join(dataset.split("/")[-2:])


def docs_generator(dataset):
    ids = []
    dataset = ir_datasets.load(dataset)
    for doc in dataset.docs_iter():
        if doc.doc_id in ids:
            continue
        ids.append(doc.doc_id)
        yield {"docno": doc.doc_id, "text": doc.default_text()}


#### IRLab ####
##### END #####


def index(dataset):
    pass    


def main():
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--dataset", help="Name or path to the dataset to be processed", required=True
    )
    args = parser.parse_args()


    if check_artefact_exist(args.dataset):
        print(f"Dataset {args.dataset} already indexed!")
        return None

    setup_index_dir(args.dataset)
    index(args.dataset)
    

if __name__ == "__main__":
    main()
