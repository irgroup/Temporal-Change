from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytrec_eval
import seaborn as sns
from adjustText import adjust_text

sns.set_style("darkgrid")


MEASURES = ["ndcg", "bpref", "P_10", "map"]
DISPLAY_NAMES = {
    "bm25": "BM25",
    "E5": "E5",
    "bm25+colbert": "ColBERT",
    "rrf(xsqram__bm25_bo1__pl2)": "RRF",
    "bm25+monot5": "MonoT5",
    "bm25_d2q10": "d2q",
}
COLORS = ["#1f78b4", "#ff7f00", "#33a02c", "#e31a1b", "#6a3c9a", "#dbdbdb"]


def plot_arp(
    df,
    dataset,
    title,
    measures=MEASURES,
    symbols=["-P", "-o", "-v", "-s", "-d"],
    sorted_columns=False,
    legend=False,
    save=False,
):
    # Data
    df = df[df["dataset"] == dataset]
    n_measures = len(measures)

    # Plot
    plt.rcParams.update({"font.size": 10})
    figure, axis = plt.subplots(1, n_measures, figsize=(15, 5))

    for n, measure in enumerate(measures):
        if sorted_columns:
            res = df.pivot_table(
                index="method", columns="subcollection", values=f"ARP_{measure}"
            )[sorted_columns].T
        else:
            res = df.pivot_table(
                index="method", columns="subcollection", values=f"ARP_{measure}"
            ).T

        for i, sym in enumerate(symbols):
            axis[n].plot(res.index, res[res.columns[i]], sym, alpha=0.7)
        axis[n].set_ylabel(measure)
        axis[n].set_title(measure)

    figure.subplots_adjust(bottom=0.3, wspace=0.33)
    figure.suptitle(title)

    if legend:
        display_name = [DISPLAY_NAMES[m] for m in res.columns]
        axis[n].legend(
            display_name, loc="upper center", bbox_to_anchor=(0.0 - 0.8, -0.15), ncol=6
        )
    if save:
        plt.savefig(f"../paper/figures/ARP_{dataset}.png", bbox_inches="tight")
    plt.show()


def plot_per_topic(
    df, dataset, method, measure, subcollections, cut_off=1000, save=False
):
    # data
    data = defaultdict(
        lambda: dict(zip(subcollections, [0 for _ in range(len(subcollections))]))
    )

    for subcollection in subcollections:
        per_topic = df[
            (df["dataset"] == dataset)
            & (df["subcollection"] == subcollection)
            & (df["method"] == method)
        ].iloc[0]["arp_per_topic"]

        for topic in per_topic.keys():
            data[topic][subcollection] = per_topic[topic][f"ARP_{measure}"]

    data = pd.DataFrame(data).T.head(cut_off)

    # Plot
    ax = data.plot(kind="bar", figsize=(30, 5))
    ax.legend()
    ax.set_ylabel(measure)
    ax.set_title(dataset)

    if save:
        plt.savefig(
            f"../../../paper/figures/ARP_per_topic_{dataset}_{method}_{measure}.png",
            bbox_inches="tight",
        )
    plt.show()


def plot_per_topic_dif(
    df, dataset, method, measure, subcollections, cut_off=1000, save=False
):
    # data
    data = defaultdict(
        lambda: dict(zip(subcollections, [0 for _ in range(len(subcollections))]))
    )
    for subcollection in subcollections:
        per_topic = df[
            (df["dataset"] == dataset)
            & (df["subcollection"] == subcollection)
            & (df["method"] == method)
        ].iloc[0]["arp_per_topic"]
        for topic in per_topic.keys():
            data[topic][subcollection] = per_topic[topic][measure]
    data = pd.DataFrame(data).T.head(cut_off)

    # plot
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.legend()
    ax.set_ylabel(measure)
    ax.set_title(dataset)

    for idx, subcollection in enumerate(subcollections):
        x = data[subcollection]
        ax.plot(x, label=subcollection, color=COLORS[idx])
        ax.fill_between(x.index, x, 0, alpha=0.2, color=COLORS[idx])

    if save:
        plt.savefig(
            f"../../../paper/figures/ARP_per_topicdiff_{dataset}_{method}_{measure}.png",
            bbox_inches="tight",
        )
    plt.show()


def plot_per_topic_delta(
    df, dataset, method, measure, subcollections, cut_off=1000, save=False
):
    assert len(subcollections) == 2, "Only two subcollections allowed"

    # data
    data = defaultdict(
        lambda: dict(zip(subcollections, [0 for _ in range(len(subcollections))]))
    )
    for subcollection in subcollections:
        per_topic = df[
            (df["dataset"] == dataset)
            & (df["subcollection"] == subcollection)
            & (df["method"] == method)
        ].iloc[0]["arp_per_topic"]

        for topic in per_topic.keys():
            data[topic][subcollection] = per_topic[topic][measure]

    data = pd.DataFrame(data).T.head(cut_off)

    data["delta"] = data[subcollections[0]] - data[subcollections[1]]

    # plot
    ax = data["delta"].sort_values(ascending=False).plot(kind="bar", figsize=(30, 5))
    ax.legend()
    ax.set_ylabel(measure)
    ax.set_title(dataset)

    if save:
        plt.savefig(
            f"../paper/figures/ARP_per_topic_delta_{dataset}_{method}_{measure}.png",
            bbox_inches="tight",
        )
    plt.show()


# Tables
def result_table(df, dataset, measure):
    custom_dict = {
        "bm25": 0,
        "E5": 5,
        "bm25+colbert": 2,
        "bm25+monot5": 3,
        "bm25_d2q10": 4,
        "rrf(xsqram__bm25_bo1__pl2)": 1,
    }
    r_patch = {
        "bm25": "BM25",
        "E5": "E5",
        "bm25+colbert": "ColBERT",
        "rrf(xsqram__bm25_bo1__pl2)": "RRF",
        "bm25+monot5": "MonoT5",
        "bm25_d2q10": "d2q",
    }

    df = df[df["dataset"] == dataset]
    columns = ["method", "subcollection"]
    columns.append(measure)

    df = df[columns]
    df = df.pivot_table(index="method", columns="subcollection", values=measure)
    df = df.reset_index()
    df["sorter"] = df["method"].replace(custom_dict)
    df = df.sort_values("sorter").drop("sorter", axis=1)
    df = df.rename_axis(None, axis=1)
    df = df.replace({"method": r_patch})
    df["measure"] = measure

    return df


def table_ARP_full(long, highlight=True, save=False):
    # data
    ARP_table = (
        long[(long["group"] == "ARP") & (long["measure"].isin(MEASURES))]
        .pivot(
            index=["measure", "method"],
            columns=["dataset", "subcollection"],
            values="value",
        )
        .sort_index(axis=1)
    )

    # style
    if highlight:

        def highlight_max(x):
            return np.where(x == np.nanmax(x), "font-weight:bold;", None)

        df_styled = ARP_table.style.format(precision=3).format_index(escape="latex")

        for index in ARP_table.index.levels[0]:
            for col in ARP_table.columns:
                df_styled = df_styled.apply(highlight_max, subset=(index, col))

        ARP_table = df_styled

    # export
    if save:
        ARP_table.to_latex(f"../paper/tables/ARP_full.tex", convert_css=True)

    return ARP_table


def table_RD_DRI(long, highlight=True, save=False):
    # data
    df = (
        long[
            (long["group"].isin(["RD", "DRI"]))
            & (long["measure"].isin(MEASURES))
            & (~long["subcollection"].isin(["WT", "t1", "round1"]))
            & (long["method"] != "BM25")
        ]
        .pivot(
            index=["measure", "method"],
            columns=["dataset", "subcollection", "group"],
            values="value",
        )
        .sort_index(axis=1)
    )

    # style
    if highlight:
        df_styled = df.style.format(precision=3).format_index(escape="latex")

        for index in df.index.levels[0]:
            for col in df.columns:
                min = df.loc[index].loc[:, col[0]].loc[:, col[1]].loc[:, col[2]].min()
                max = df.loc[index].loc[:, col[0]].loc[:, col[1]].loc[:, col[2]].max()

                df_styled = df_styled.background_gradient(
                    cmap="Greens", subset=(index, col), vmin=min, vmax=max
                )
        df = df_styled

    # export
    if save:
        df.to_latex(f"../paper/tables/RD_DRI.tex", convert_css=True)

    return df


def _plot_DRI_ER(long, dataset, subcollection, measures=MEASURES, save=False):
    marker_color = [("o", "#1f78b4"), ("^", "#33a02c"), ("v", "#e31a1b")]

    fig, ax = plt.subplots(figsize=(7, 7))

    methods = long["method"].unique()

    for measure, mk in zip(measures, marker_color):
        # data
        er = long[
            (long["dataset"] == dataset)
            & (long["measure"] == measure)
            & (long["group"] == "ER")
            & (long["subcollection"] == subcollection)
        ]["value"].clip(0, 2)
        dri = long[
            (long["dataset"] == dataset)
            & (long["measure"] == measure)
            & (long["group"] == "DRI")
            & (long["subcollection"] == subcollection)
        ]["value"].clip(-0.1, 0.1)

        # plot
        ax.plot(er, dri, marker=mk[0], color=mk[1], linestyle="None", label=measure)

    # label
    texts = []
    for measure in measures:
        for method in methods:
            texts.append(
                ax.text(
                    x=long[
                        (long["dataset"] == dataset)
                        & (long["measure"] == measure)
                        & (long["group"] == "ER")
                        & (long["subcollection"] == subcollection)
                        & (long["method"] == method)
                    ]["value"]
                    .clip(0, 2)
                    .values[0],
                    y=long[
                        (long["dataset"] == dataset)
                        & (long["measure"] == measure)
                        & (long["group"] == "DRI")
                        & (long["subcollection"] == subcollection)
                        & (long["method"] == method)
                    ]["value"]
                    .clip(-0.1, 0.1)
                    .values[0],
                    s=method,
                )
            )

    # style
    ax.tick_params(axis="y", labelcolor="k")
    ax.axhline(0, color="grey")
    ax.axvline(1, color="grey")

    ax.grid(False)  # disable grid
    ax.set_xlabel("Effect Ratio (ER)")
    ax.set_ylabel("Delta Relative Improvement ($\Delta$RI)")

    ax.set_ylim(-0.1, 0.1)
    ax.set_xlim(0, 2)

    ax.legend(loc="lower left")
    ax.set_title(f"{dataset} {subcollection}\n")

    _ = adjust_text(texts, ax=ax)

    if save:
        plt.savefig(
            f"../paper/figures/ER_DRI_{dataset}-{subcollection}.png",
            bbox_inches="tight",
        )
    plt.show()


def plot_DRI_ER(long):
    for dataset in long["dataset"].unique():
        for subcollection in long[long["dataset"] == dataset]["subcollection"].unique():
            if subcollection in ["WT", "round1", "t1"]:
                continue
            _plot_DRI_ER(long, dataset, subcollection, save=False)


def plot_arp_all(df, measure="bpref", save=False):
    # Plot
    plt.rcParams.update({"font.size": 13})
    figure, axis = plt.subplots(
        1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1.3]}
    )

    symbols = ["-P", "-o", "-v", "-s", "-d"]

    # LongEval
    dataset = "longeval"
    title = "LongEval"

    res = df[df["dataset"] == dataset]
    sorted_columns = ["WT", "ST", "LT"]

    res = res.pivot_table(
        index="method", columns="subcollection", values=f"ARP_{measure}"
    )[sorted_columns]
    res = res.rename(columns={"WT": "$t_0$", "ST": "$t_1$", "LT": "$t_2$"})
    res = res.T

    for i, sym in enumerate(symbols):
        axis[0].plot(res.index, res[res.columns[i]], sym, alpha=0.7)
    axis[0].set_ylabel("bpref")
    axis[0].set_title(title)
    axis[0].set_ylim(0.3, 0.46)

    # TripClick
    dataset = "tripclick-test-head"
    title = "TripClick"

    res = df[df["dataset"] == dataset]

    res = res.pivot_table(
        index="method", columns="subcollection", values=f"ARP_{measure}"
    )
    res = res.rename(columns={"t1": "$t_0$", "t2": "$t_1$", "t3": "$t_2$"})
    res = res.T

    for i, sym in enumerate(symbols):
        axis[1].plot(res.index, res[res.columns[i]], sym, alpha=0.7)
    axis[1].set_title(title)
    axis[1].set_ylim(0.3, 0.46)

    # TREC-COVID
    dataset = "trec-covid"
    title = "TREC-COVID"

    res = df[df["dataset"] == dataset]

    res = res.pivot_table(
        index="method", columns="subcollection", values=f"ARP_{measure}"
    )
    res = res.rename(
        columns={
            "round1": "$t_0$",
            "round2": "$t_1$",
            "round3": "$t_2$",
            "round4": "$t_3$",
            "round5": "$t_4$",
        }
    )
    res = res.T

    for i, sym in enumerate(symbols):
        axis[2].plot(res.index, res[res.columns[i]], sym, alpha=0.7)
    axis[2].set_title(title)
    axis[2].set_ylim(0.3, 0.46)

    figure.subplots_adjust(bottom=0.25, wspace=0.33)

    display_name = [DISPLAY_NAMES[m] for m in res.columns]

    figure.text(0.5, 0.14, "Time", ha="center")

    figure.legend(display_name, loc="lower center", ncol=6)
    # figure.suptitle(title)

    if save:
        plt.savefig(f"../paper/figures/ARP_all.png", bbox_inches="tight", dpi=700)
