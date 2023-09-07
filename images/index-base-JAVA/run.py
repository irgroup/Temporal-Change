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


def fix_ir_dataset_naming(dataset_name):
    return "-".join(dataset_name.split("/")[-2:])


def docs_generator(dataset_name):
    ids = []
    dataset = ir_datasets.load(dataset_name)
    for doc in dataset.docs_iter():
        if doc.doc_id in ids:
            continue
        ids.append(doc.doc_id)
        yield {"docno": doc.doc_id, "text": doc.default_text()}


#### IRLab ####
##### END #####


def index(dataset_name):
    pass    


def main():
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", help="Name or path to the dataset to be processed", required=True
    )
    args = parser.parse_args()


    if check_artefact_exist(args.dataset_name):
        print(f"Dataset {args.dataset_name} already indexed!")
        return None

    setup_index_dir(args.dataset_name)
    index(args.dataset_name)
    

if __name__ == "__main__":
    main()
