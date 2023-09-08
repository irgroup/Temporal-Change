import os
from argparse import ArgumentParser

from system import rank

SYSTEM = os.getenv("SYSTEM")
METHOD = os.getenv("METHOD")


def index_name(index):
    return "-".join(index.split("/")[-2:][0].split("-")[1:-1])


def query_name(query):
    return query.split("/")[-1].split(".")[0]


def run_path(index, query):
    index = index_name(index)
    query = query_name(query)
    return f"./data/run/run-{index}-{query}-{SYSTEM}-{METHOD}"


def resolve_queries(index, queries):
    if queries == "queries":
        return index.replace("/documents", "/queries/queries.trec")


def main():
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--index", help="Name or path to the dataset to be processed", required=True
    )
    parser.add_argument(
        "--queries", help="Name of the query file", required=False, default="queries"
    )

    args = parser.parse_args()

    queries = resolve_queries(args.index, args.queries)
    result_path = run_path(args.index, queries)

    rank(args.index, queries, result_path)


if __name__ == "__main__":
    main()
