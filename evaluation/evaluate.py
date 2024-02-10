import os

import numpy as np
import pandas as pd
import pytrec_eval
from repro_eval.Evaluator import RpdEvaluator, RplEvaluator
from repro_eval.util import arp, arp_scores

MEASURES = ["ndcg", "bpref", "P_10"]
DISPLAY_NAMES = {
    "bm25": "BM25",
    "E5": "E5",
    "bm25+colbert": "ColBERT",
    "rrf(xsqram__bm25_bo1__pl2)": "RRF",
    "bm25+monot5": "MonoT5",
    "bm25_d2q10": "d2q",
}


def load_runs_metadata_table(mode=""):
    table = []
    for run in os.listdir(f"../data/run{mode}"):
        parts = run.split("-")
        fields = {
            "dataset": "-".join(parts[1:-4]),
            "subcollection": parts[-4],
            "queries": parts[-3],
            "method": parts[-2],
            "implementation": parts[-1],
            "filename": run,
        }
        table.append(fields)
    runs = pd.DataFrame(table)
    # runs = runs[
    #     ~((runs["subcollection"] == "WT") & (runs["queries"] != "test"))
    # ]  # longeval WT test only
    runs = runs[
        ~((runs["subcollection"] == "WT") & (runs["queries"] != "queries"))
    ]  # longeval WT all
    runs = runs[
        runs["method"].isin(
            [
                "bm25",
                "bm25+colbert",
                "bm25+monot5",
                "rrf(xsqram__bm25_bo1__pl2)",
                "bm25_d2q10",
            ]
        )
    ]
    return runs


# ARP
def get_qrels_name_from_row(row):
    qrels_type = {"tripclick-test-head": "test-head-dctr"}  # tripclick needs mapping

    qrels_name = row.dataset + "-" + row.subcollection + ".qrels"
    if row.queries != "queries":
        qrels_name += "-" + row.queries
    if row.dataset in qrels_type.keys():
        qrels_name += "-" + qrels_type[row.dataset]

    return qrels_name


def load_evaluator_with_qrels(qrels_name, mode=""):
    with open("../data/qrels/" + qrels_name + mode, "r") as file:
        qrels = pytrec_eval.parse_qrel(file)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, pytrec_eval.supported_measures)
    return evaluator


def _evaluate_arp(row, mode):
    evaluator = load_evaluator_with_qrels(get_qrels_name_from_row(row), mode=mode)

    with open(f"../data/run{mode}/" + row.filename, "r") as file:
        run = pytrec_eval.parse_run(file)

    arp = evaluator.evaluate(run)
    return arp


def evaluate_arp(table, mode=""):
    table["arp_per_topic"] = table.apply(_evaluate_arp, mode=mode, axis=1)
    table["arp"] = table["arp_per_topic"].apply(arp_scores)
    table = pd.concat(
        [table.drop(["arp"], axis=1), table["arp"].apply(pd.Series).add_prefix("ARP_")],
        axis=1,
    )
    return table


def is_reprocuction(row):
    # if row["subcollection"] not in ["WT", "t1", "round1"] and row["method"] != "bm25":
    if row["subcollection"] not in ["WT", "t1", "round1"]:
        return True
    else:
        return False


# Result delta
def evaluate_result_delta(table):
    def calc_result_delta(base, advanced):
        return (base - advanced) / base

    def _evaluate_result_delta(row):
        metadata_columns = [
            "dataset",
            "subcollection",
            "queries",
            "method",
            "implementation",
            "filename",
            "arp_per_topic",
        ]

        if is_reprocuction(row):
            result_delta = {}

            # get base run
            base_run = table[
                (table["method"] == row["method"])
                & (table["dataset"] == row["dataset"])
                & (table["subcollection"].isin(["t1", "WT", "round1"]))
            ].iloc[0]

            # calculate delta
            for measure in table.columns:
                if measure not in metadata_columns:
                    delta = calc_result_delta(base_run[measure], row[measure])
                    result_delta["RD_" + measure.replace("ARP_", "")] = delta
            return result_delta

        else:
            return None

    table["result_delta"] = table.apply(_evaluate_result_delta, axis=1)
    table = pd.concat(
        [
            table.drop(["result_delta"], axis=1),
            table["result_delta"].apply(pd.Series),
        ],
        axis=1,
    )
    return table


# Replicability
def evaluate_replicability(table, mode=""):
    def _evaluate_replicability(row, mode):
        # This fails if topics are in the advanced run but not in the baseline run. This is the case if the baseline run retrieves 0 docs for a topic but an advanced system retrieves any, for example, because it bridges the lexical gap.
        if is_reprocuction(row):
            try:
                # Original
                # get qrel baseline
                qrel_orig_path = get_qrels_name_from_row(
                    runs_b_orig[runs_b_orig["dataset"] == row["dataset"]].iloc[0]
                )
                # baseline
                run_b_orig_path = runs_b_orig[
                    runs_b_orig["dataset"] == row["dataset"]
                ].iloc[0]["filename"]
                # advanced
                run_a_orig_path = runs_a_orig[
                    (runs_a_orig["dataset"] == row["dataset"])
                    & (runs_a_orig["method"] == row["method"])
                ].iloc[0]["filename"]

                # Replicated
                # get baseline
                run_b_rep_path = runs_b_rep[
                    runs_b_rep["subcollection"] == row["subcollection"]
                ].iloc[0]["filename"]
                # get qrel advanced
                qrel_rpl_path = get_qrels_name_from_row(row)

                rpl_eval = RplEvaluator(
                    qrel_orig_path="../data/qrels/" + qrel_orig_path + mode,
                    run_b_orig_path=f"../data/run{mode}/" + run_b_orig_path,
                    run_a_orig_path=f"../data/run{mode}/" + run_a_orig_path,
                    run_b_rep_path=f"../data/run{mode}/" + run_b_rep_path,
                    run_a_rep_path=f"../data/run{mode}/" + row["filename"],
                    qrel_rpl_path="../data/qrels/" + qrel_rpl_path + mode,
                )

                rpl_eval.trim()
                rpl_eval.evaluate()
                ret = {
                    "dri": rpl_eval.dri(),
                    "er": rpl_eval.er(),
                    "pval": rpl_eval.ttest(),
                }
            except:
                print(
                    "Error in",
                    row["filename"],
                    "not all topics match with BM25 baseline",
                )
                ret = {"dri": np.nan, "er": np.nan, "pval": np.nan}
            return ret
        else:
            return {"dri": np.nan, "er": np.nan, "pval": np.nan}

    # group runs
    runs_b_orig = table[
        (table["subcollection"].isin(["WT", "t1", "round1"]))
        & (table["method"] == "bm25")
    ]
    runs_a_orig = table[
        (table["subcollection"].isin(["WT", "t1", "round1"]))
        & (table["method"] != "bm25")
    ]

    runs_b_rep = table[
        (~table["subcollection"].isin(["WT", "t1", "round1"]))
        & (table["method"] == "bm25")
    ]
    runs_a_rep = table[
        (~table["subcollection"].isin(["WT", "t1", "round1"]))
        & (table["method"] != "bm25")
    ]

    # evaluate replication
    table["replicability"] = table.apply(_evaluate_replicability, mode=mode, axis=1)

    # Explode table
    table = pd.concat(
        [
            table.drop(["replicability"], axis=1),
            table["replicability"].apply(pd.Series),
        ],
        axis=1,
    )
    table = pd.concat(
        [table.drop(["dri"], axis=1), table["dri"].apply(pd.Series).add_prefix("DRI_")],
        axis=1,
    )
    table = pd.concat(
        [table.drop(["er"], axis=1), table["er"].apply(pd.Series).add_prefix("ER_")],
        axis=1,
    )
    table = pd.concat(
        [
            table.drop(["pval"], axis=1),
            table["pval"].apply(pd.Series).add_prefix("PVAL_"),
        ],
        axis=1,
    )
    table = pd.concat(
        [
            table.drop(["PVAL_advanced"], axis=1),
            table["PVAL_advanced"].apply(pd.Series).add_prefix("PVAL_"),
        ],
        axis=1,
    )
    return table


# Replicability
def evaluate_reproducibility(table, mode=""):
    cutoffs = [100, 50, 20, 10, 5]  # max 281

    def _evaluate_reproducibility(row, mode):
        # This fails if topics are in the advanced run but not in the baseline run. This is the case if the baseline run retrieves 0 docs for a topic but an advanced system retrieves any, for example, because it bridges the lexical gap.
        if is_reprocuction(row):
            # Original
            # get qrel baseline
            qrel_orig_path = get_qrels_name_from_row(
                runs_b_orig[runs_b_orig["dataset"] == row["dataset"]].iloc[0]
            )
            # baseline
            run_b_orig_path = runs_b_orig[
                runs_b_orig["dataset"] == row["dataset"]
            ].iloc[0]["filename"]
            # advanced
            run_a_orig_path = runs_a_orig[
                (runs_a_orig["dataset"] == row["dataset"])
                & (runs_a_orig["method"] == row["method"])
            ].iloc[0]["filename"]

            # Replicated
            # get baseline
            run_b_rep_path = runs_b_rep[
                runs_b_rep["subcollection"] == row["subcollection"]
            ].iloc[0]["filename"]
            # get qrel advanced
            qrel_rpl_path = get_qrels_name_from_row(row)

            rpd_eval = RpdEvaluator(
                qrel_orig_path="../data/qrels/" + qrel_orig_path + mode,
                run_b_orig_path=f"../data/run{mode}/" + run_b_orig_path,
                run_a_orig_path=f"../data/run{mode}/" + run_a_orig_path,
                run_b_rep_path=f"../data/run{mode}/" + run_b_rep_path,
                run_a_rep_path=f"../data/run{mode}/" + row["filename"],
                qrel_rpl_path="../data/qrels/" + qrel_rpl_path + mode,
            )

            try:
                rpd_eval.trim()
                rpd_eval.evaluate()
                ret = {"rmse": rpd_eval.rmse()}
            except:
                ret["rmse"] = {}
                print("Error RMSE", row["filename"])

            try:
                ret["rbo"] = {}
                for cutoff in cutoffs:
                    rpd_eval.trim(t=cutoff)
                    rpd_eval.evaluate()
                    ret["rbo"][f"rbo_{cutoff}"] = arp(rpd_eval.rbo()["advanced"])
            except:
                ret["rbo"] = {}
                print("Error RBO", row["filename"])

            try:
                ret["ktau"] = {}
                for cutoff in cutoffs:
                    rpd_eval.trim(t=cutoff)
                    rpd_eval.evaluate()
                    ret["ktau"][f"ktau_{cutoff}"] = arp(
                        rpd_eval.ktau_union()["advanced"]
                    )
            except:
                ret["ktau"] = {}
                print("Error KTau", row["filename"])

            return ret
        else:
            return {"rmse": np.nan, "rbo": np.nan}

    # group runs
    runs_b_orig = table[
        (table["subcollection"].isin(["WT", "t1", "round1"]))
        & (table["method"] == "bm25")
    ]
    # runs_a_orig = table[
    #     (table["subcollection"].isin(["WT", "t1", "round1"]))
    #     & (table["method"] != "bm25")
    # ]
    runs_a_orig = table[(table["subcollection"].isin(["WT", "t1", "round1"]))]

    runs_b_rep = table[
        (~table["subcollection"].isin(["WT", "t1", "round1"]))
        & (table["method"] == "bm25")
    ]
    # runs_a_rep = table[
    #     (~table["subcollection"].isin(["WT", "t1", "round1"]))
    #     & (table["method"] != "bm25")
    # ]
    runs_a_rep = table[(~table["subcollection"].isin(["WT", "t1", "round1"]))]

    # evaluate replication
    table["reproducibility"] = table.apply(_evaluate_reproducibility, mode=mode, axis=1)

    # Explode table
    table = pd.concat(
        [
            table.drop(["reproducibility"], axis=1),
            table["reproducibility"].apply(pd.Series),
        ],
        axis=1,
    )
    table = pd.concat(
        [table.drop(["rbo"], axis=1), table["rbo"].apply(pd.Series).add_prefix("RBO_")],
        axis=1,
    )
    # table = pd.concat(
    #     [table.drop(["RBO_advanced"], axis=1), table["RBO_advanced"].apply(pd.Series).add_prefix("RBO_")],
    #     axis=1,
    # )
    table = pd.concat(
        [
            table.drop(["rmse"], axis=1),
            table["rmse"].apply(pd.Series).add_prefix("RMSE_"),
        ],
        axis=1,
    )
    table = pd.concat(
        [
            table.drop(["RMSE_advanced"], axis=1),
            table["RMSE_advanced"].apply(pd.Series).add_prefix("RMSE_"),
        ],
        axis=1,
    )
    return table


def table_ARP_statsig(df, dataset, subcollections, highlight=True, save=False, mode=""):
    import pyterrier as pt

    if not pt.started():
        pt.init()

    def mark_stat_sig(row):
        for measure in MEASURES:
            if row[f"{measure} reject"]:
                score = row[measure]
                row[measure] = str(score) + "*"
        return row

    # data
    sort_dict = {
        "bm25": 0,
        "E5": 5,
        "bm25+colbert": 2,
        "bm25+monot5": 3,
        "bm25_d2q10": 4,
        "rrf(xsqram__bm25_bo1__pl2)": 1,
    }
    topics_paths = {
        "longeval": "../data/index/index-{dataset}-{subcollection}-pyterrier/queries/test.trec",
        "trec-covid": "../data/index/index-{dataset}-{subcollection}-pyterrier/queries/queries.trec",
        "tripclick-test-head": "../data/index/index-{dataset}-{subcollection}-pyterrier/queries/queries.trec",
    }

    results = []
    for subcollection in subcollections:
        runs_table = df[
            (df["dataset"] == dataset) & (df["subcollection"] == subcollection)
        ]
        runs_table["sorter"] = runs_table["method"].replace(sort_dict)
        runs_table = runs_table.sort_values("sorter").drop("sorter", axis=1)

        runs = []
        names = []
        for _, row in runs_table.iterrows():
            runs.append(pt.io.read_results(f"../data/run{mode}/" + row["filename"]))
            names.append(row["method"])

        # qrels
        qrels_path = (
            "../data/qrels/" + get_qrels_name_from_row(runs_table.iloc[0]) + mode
        )
        qrels = pt.io.read_qrels(qrels_path)

        # topics
        topics_path = topics_paths[dataset].format(
            dataset=dataset, subcollection=subcollection
        )
        if dataset == "longeval" and subcollection == "WT":
            topics_path = "../data/index/index-longeval-WT-pyterrier/queries/full.trec"

        if mode and dataset == "longeval":
            topics = pt.io.read_topics(
                "longeval_topics_core_queries_unified.tsv", format="singleline"
            )
        else:
            topics = pt.io.read_topics(topics_path)

        # evaluation
        res = pt.Experiment(
            retr_systems=runs,
            names=names,
            topics=topics,
            qrels=qrels,
            eval_metrics=MEASURES,
            baseline=0,
            correction="bonferroni",
            filter_by_qrels=True,
        )

        # table
        res = res.replace({"name": DISPLAY_NAMES})
        res = res.round(3)
        res = res.apply(mark_stat_sig, axis=1)

        columns = ["name"]
        columns.extend(MEASURES)
        res = res[columns]
        res["subcollection"] = subcollection
        results.append(res)

    # merge tables
    res = pd.concat(results)
    table = []
    for measure in MEASURES:
        res_pivot = res.pivot(index="name", columns="subcollection", values=measure)
        res_pivot["measure"] = measure
        res_pivot = res_pivot.set_index(["measure", res_pivot.index])
        table.append(res_pivot)
    ret = pd.concat(table)
    ret = ret[subcollections]
    ret = ret.round(3)

    if highlight:

        def highlight_max(x):
            a = []
            for i in x:
                if isinstance(i, str):
                    a.append(float(i[:-1]))
                else:
                    a.append(i)
            return np.where(a == np.nanmax(a), "font-weight:bold;", None)

        df_styled = ret.style.format(precision=3).format_index(escape="latex")

        for measure in ret.index.levels[0]:
            for subcollection in ret.columns:
                df_styled = df_styled.apply(
                    highlight_max, subset=(measure, subcollection)
                )

        ret = df_styled

        if save:
            ret.to_latex(
                f"../../../paper/ECIR23/tables/ARP_{dataset}.tex", convert_css=True
            )

    return ret


def long_table(df):
    long = df.drop(columns=["implementation", "filename", "arp_per_topic", "queries"])
    long = pd.melt(long, id_vars=["method", "subcollection", "dataset"])

    long["group"] = long["variable"].str.split("_", n=1).str[0]
    long["measure"] = long["variable"].str.split("_", n=1).str[1]

    long = long[long["measure"] != "baseline"]
    long = long[long["measure"] != "0"]

    long["value"] = long["value"].apply(
        lambda x: round(x, 3) if isinstance(x, float) else x
    )
    long = long.replace({"method": DISPLAY_NAMES})
    return long
