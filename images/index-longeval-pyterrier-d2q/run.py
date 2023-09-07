from argparse import ArgumentParser
from indexer import index
import os


METHOD = os.getenv("METHOD")
TYPE = os.getenv("TYPE")


def get_artefact_path(artefact_name, num_samples):
    artefact_type = TYPE
    if artefact_type == "index":
        return (f"/data/index/index-{artefact_name}-{METHOD}-d2q{num_samples}")
    elif artefact_type == "run":
        return f"/data/runs/"
    elif artefact_type == "model":
        return f"/data/models/model-{artefact_name}-{METHOD}-d2q{num_samples}"
    else:
        raise ValueError(f"Artefact type `{artefact_type}` not supported!")


def setup_index_dir(dataset_name, num_samples):
    index_path = get_artefact_path(dataset_name, num_samples)
    os.makedirs(f"{index_path}/documents")
    os.makedirs(f"{index_path}/queries")


def main():
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", help="Name or path to the dataset to be processed", required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for d2q",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Num samples to extend the document with",
    )

    args = parser.parse_args()

    setup_index_dir(args.dataset_name, args.num_samples)

    index_document_path = get_artefact_path(args.dataset_name, args.num_samples) + "/documents"
    index_query_path = get_artefact_path(args.dataset_name, args.num_samples) + "/queries"

    index(args.dataset_name, index_document_path, index_query_path, args.batch_size, args.num_samples)

if __name__ == "__main__":
    main()
