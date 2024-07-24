import os

import numpy as np
import pandas as pd


def generate_plot_dro(
    base_path,
    timestamp,
    ax,
    empirical_field="empirical_conditional_mean",
    true_field="true_conditional_mean",
):
    plotting_df = pd.read_csv(
        os.path.join(base_path, timestamp, "plotting_df.csv"), low_memory=False
    )
    max_steps = plotting_df.groupby(["matrix_type", "restriction_type"]).max()

    masks = {
        ("unrestricted", "DRO"): (
            plotting_df["step"] == max_steps.loc[("unrestricted", "DRO"), "step"]
        )
        & (plotting_df["restriction_type"] == "DRO"),
    }

    plot_intervals(
        ax=ax, masks=masks, plotting_df=plotting_df, color="C0", mode="single_dro"
    )

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field=empirical_field,
        true_field=true_field,
    )
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_xticks([1])
    ax.set_xticklabels(["Omniscient DRO"], rotation=45)
    return plotting_df


def generate_constraints_plots(
    base_path: str, timestamp_ours: str, timestamp_dro: str, ax
):
    plotting_df_ours = pd.read_csv(
        os.path.join(base_path, timestamp_ours, "plotting_df.csv"), low_memory=False
    )
    plotting_df_dro = pd.read_csv(
        os.path.join(base_path, timestamp_dro, "plotting_df.csv"), low_memory=False
    )

    plotting_df = pd.concat([plotting_df_ours, plotting_df_dro], axis=0)

    max_steps = plotting_df.groupby(["matrix_type", "restriction_type"]).max()

    masks = {
        ("unrestricted", "DRO_worst_case"): (
            plotting_df["step"]
            == max_steps.loc[("unrestricted", "DRO_worst_case"), "step"]
        )
        & (plotting_df["matrix_type"] == "unrestricted")
        & (plotting_df["restriction_type"] == "DRO_worst_case"),
        ("unrestricted", "count"): (
            plotting_df["step"] == max_steps.loc[("unrestricted", "count"), "step"]
        )
        & (plotting_df["matrix_type"] == "unrestricted")
        & (plotting_df["restriction_type"] == "count"),
        # ("separable", "DRO_worst_case"): (
        #     plotting_df["step"]
        #     == max_steps.loc[("separable", "DRO_worst_case"), "step"]
        # )
        # & (plotting_df["matrix_type"] == "separable")
        # & (plotting_df["restriction_type"] == "DRO_worst_case"),
        # ("separable", "count"): (
        #     plotting_df["step"] == max_steps.loc[("separable", "count"), "step"]
        # )
        # & (plotting_df["matrix_type"] == "separable")
        # & (plotting_df["restriction_type"] == "count"),
        # ("targeted", "DRO_worst_case"): (
        #     plotting_df["step"] == max_steps.loc[("targeted", "DRO_worst_case"), "step"]
        # )
        # & (plotting_df["matrix_type"] == "targeted")
        # & (plotting_df["restriction_type"] == "DRO_worst_case"),
        # ("targeted", "count"): (
        #     plotting_df["step"] == max_steps.loc[("targeted", "count"), "step"]
        # )
        # & (plotting_df["matrix_type"] == "targeted")
        # & (plotting_df["restriction_type"] == "count"),
    }

    plot_intervals(ax=ax, masks=masks, plotting_df=plotting_df, color="C0")
    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_conditional_mean",
        true_field="true_conditional_mean",
    )
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels(["Unrestricted", "Separable", "Targeted"], rotation=45)
    ax.set_ylabel("Conditional Mean $\widehat{\mathbb{E}}[Y|A=1]$", fontsize=30)
    ax.set_xlabel("Parametric form of $\\theta(X)$", fontsize=30)
    return plotting_df


def generate_theta_plots(base_path: str, timestamp_ours: str, timestamp_dro: str, ax):
    plotting_df_ours = pd.read_csv(
        os.path.join(base_path, timestamp_ours, "plotting_df.csv"), low_memory=False
    )
    plotting_df_dro = pd.read_csv(
        os.path.join(base_path, timestamp_dro, "plotting_df.csv"), low_memory=False
    )

    plotting_df_ours["dro_restriction_type"] = plotting_df_ours["restriction_type"]

    plotting_df = pd.concat([plotting_df_ours, plotting_df_dro], axis=0)

    max_steps = plotting_df.groupby(["restriction_type", "dro_restriction_type"]).max()

    masks = {
        ("DRO_worst_case", "count_minus"): (
            plotting_df["step"]
            == max_steps.loc[("DRO_worst_case", "count_minus"), "step"]
        )
        & (plotting_df["restriction_type"] == "DRO_worst_case")
        & (plotting_df["dro_restriction_type"] == "count_minus"),
        ("count_minus", "count_minus"): (
            plotting_df["step"] == max_steps.loc[("count_minus", "count_minus"), "step"]
        )
        & (plotting_df["restriction_type"] == "count_minus")
        & (plotting_df["dro_restriction_type"] == "count_minus"),
        ("DRO_worst_case", "count"): (
            plotting_df["step"] == max_steps.loc[("DRO_worst_case", "count"), "step"]
        )
        & (plotting_df["restriction_type"] == "DRO_worst_case")
        & (plotting_df["dro_restriction_type"] == "count"),
        ("count", "count"): (
            plotting_df["step"] == max_steps.loc[("count", "count"), "step"]
        )
        & (plotting_df["restriction_type"] == "count")
        & (plotting_df["dro_restriction_type"] == "count"),
        ("DRO_worst_case", "count_plus"): (
            plotting_df["step"]
            == max_steps.loc[("DRO_worst_case", "count_plus"), "step"]
        )
        & (plotting_df["restriction_type"] == "DRO_worst_case")
        & (plotting_df["dro_restriction_type"] == "count_plus"),
        ("count_plus", "count_plus"): (
            plotting_df["step"] == max_steps.loc[("count_plus", "count_plus"), "step"]
        )
        & (plotting_df["restriction_type"] == "count_plus")
        & (plotting_df["dro_restriction_type"] == "count_plus"),
    }

    plot_intervals(ax=ax, masks=masks, plotting_df=plotting_df, color="C0")

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_conditional_mean",
        true_field="true_conditional_mean",
    )
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels(
        [
            "(Partial) Race\n+ Income",
            "(Full) Race\n+ Income",
            "Race\n+ Income\n+ Outcome",
        ],
        rotation=45,
        fontsize=18,
    )

    ax.set_xlabel("Number of constraints in $\\theta(X)$", fontsize=30)
    return plotting_df


def generate_cov_plots(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))
    max_steps = plotting_df.groupby(["restriction_type"]).max()

    masks = {
        "cov_positive": (plotting_df["step"] == max_steps.loc["cov_positive", "step"])
        & (plotting_df["restriction_type"] == "cov_positive"),
        "count": (plotting_df["step"] == max_steps.loc["count", "step"])
        & (plotting_df["restriction_type"] == "count"),
    }

    plot_intervals(ax=ax, masks=masks, plotting_df=plotting_df, color="C0", mode="cov")

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_conditional_mean",
        true_field="true_conditional_mean",
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Positive Covariance", "Targeted"])
    ax.set_ylabel("Conditional Mean $\widehat{\mathbb{E}}[Y|A=1]$", fontsize=20)
    ax.set_xlabel("Constraint Type", fontsize=20)

    ax.legend(loc="lower right", fontsize=18)
    return plotting_df


def generate_constraints_plots_regression(
    base_path: str, timestamp_ours: str, timestamp_dro: str, ax
):
    plotting_df_ours = pd.read_csv(
        os.path.join(base_path, timestamp_ours, "plotting_df.csv")
    )
    plotting_df_dro = pd.read_csv(
        os.path.join(base_path, timestamp_dro, "plotting_df.csv")
    )

    plotting_df_ours["dro_restriction_type"] = plotting_df_ours["restriction_type"]
    plotting_df = pd.concat([plotting_df_ours, plotting_df_dro], axis=0)
    max_steps = plotting_df.groupby(["matrix_type", "restriction_type"]).max()

    masks = {
        ("unrestricted", "DRO_worst_case"): (
            plotting_df["step"]
            == max_steps.loc[("unrestricted", "DRO_worst_case"), "step"]
        )
        & (plotting_df["matrix_type"] == "unrestricted")
        & (plotting_df["restriction_type"] == "DRO_worst_case"),
        ("unrestricted", "count"): (
            plotting_df["step"] == max_steps.loc[("unrestricted", "count"), "step"]
        )
        & (plotting_df["matrix_type"] == "unrestricted")
        & (plotting_df["restriction_type"] == "count"),
        ("targeted", "DRO_worst_case"): (
            plotting_df["step"] == max_steps.loc[("targeted", "DRO_worst_case"), "step"]
        )
        & (plotting_df["matrix_type"] == "targeted")
        & (plotting_df["restriction_type"] == "DRO_worst_case"),
        ("targeted", "count"): (
            plotting_df["step"] == max_steps.loc[("targeted", "count"), "step"]
        )
        & (plotting_df["matrix_type"] == "targeted")
        & (plotting_df["restriction_type"] == "count"),
    }

    plot_intervals(
        ax=ax, masks=masks, plotting_df=plotting_df, color="C0", mode="dro_comparison"
    )

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_coef",
        true_field="true_coef",
    )
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.set_xticks([1, 3])
    ax.set_xticklabels(["Unrestricted", "Targeted"], rotation=45)
    ax.set_ylabel("Estimated Coefficient", fontsize=26)
    ax.set_xlabel("Parametric form of $\\theta(X)$", fontsize=26)
    return plotting_df


def generate_theta_plots_regression(
    base_path: str, timestamp_ours: str, timestamp_dro: str, ax
):
    plotting_df_ours = pd.read_csv(
        os.path.join(base_path, timestamp_ours, "plotting_df.csv")
    )
    plotting_df_dro = pd.read_csv(
        os.path.join(base_path, timestamp_dro, "plotting_df.csv")
    )
    plotting_df_ours["dro_restriction_type"] = plotting_df_ours["restriction_type"]
    plotting_df = pd.concat([plotting_df_ours, plotting_df_dro], axis=0)
    max_steps = plotting_df.groupby(["restriction_type", "dro_restriction_type"]).max()

    masks = {
        ("DRO_worst_case", "count"): (
            plotting_df["step"] == max_steps.loc[("DRO_worst_case", "count"), "step"]
        )
        & (plotting_df["restriction_type"] == "DRO_worst_case")
        & (plotting_df["dro_restriction_type"] == "count"),
        ("count", "count"): (
            plotting_df["step"] == max_steps.loc[("count", "count"), "step"]
        )
        & (plotting_df["restriction_type"] == "count")
        & (plotting_df["dro_restriction_type"] == "count"),
        ("DRO_worst_case", "count_plus"): (
            plotting_df["step"]
            == max_steps.loc[("DRO_worst_case", "count_plus"), "step"]
        )
        & (plotting_df["restriction_type"] == "DRO_worst_case")
        & (plotting_df["dro_restriction_type"] == "count_plus"),
        ("count_plus", "count_plus"): (
            plotting_df["step"] == max_steps.loc[("count_plus", "count_plus"), "step"]
        )
        & (plotting_df["restriction_type"] == "count_plus")
        & (plotting_df["dro_restriction_type"] == "count_plus"),
    }

    plot_intervals(
        ax=ax, masks=masks, plotting_df=plotting_df, color="C0", mode="dro_comparison"
    )

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_coef",
        true_field="true_coef",
    )
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.set_xticks([1, 3])
    ax.set_xticklabels(
        [
            "(Full) Race\n+ Income",
            "Race\n+ Income\n+ Outcome",
        ],
        rotation=45,
        fontsize=18,
    )
    ax.set_xlabel("Number of constraints in $\\theta(X)$", fontsize=26)
    return plotting_df


def generate_logistic_plot(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))
    max_steps = plotting_df.groupby(["restriction_type"]).max()

    masks = {
        "count_minus": (plotting_df["step"] == max_steps.loc["count_minus", "step"])
        & (plotting_df["restriction_type"] == "count_minus"),
    }

    plot_intervals(
        ax=ax,
        masks=masks,
        plotting_df=plotting_df,
        color="C0",
        mode="logistic_regression",
    )

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_coef",
        true_field="true_coef",
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([1])
    ax.set_xticklabels(["(Partial) Race\n+ Income"])
    ax.set_ylabel("Estimated Coefficient", fontsize=26)
    ax.set_xlabel("Number of constraints in $\\theta(X)$", fontsize=20)

    return plotting_df


def plot_intervals(
    ax, masks: dict, plotting_df: pd.DataFrame, color="C0", mode="dro_comparison"
):
    experiment_bounds = {}
    for name, mask in masks.items():
        experiment_bounds[name] = {
            "max": plotting_df[mask]["max_bound"],
            "min": plotting_df[mask]["min_bound"],
        }
    if mode == "dro_comparison":
        centers = {
            1: 0.75,
            2: 1.25,
            3: 2.75,
            4: 3.25,
            5: 4.75,
            6: 5.25,
        }
    else:
        centers = {idx: idx for idx in range(1, len(experiment_bounds) + 1)}
    for idx, (name, bounds) in enumerate(experiment_bounds.items(), start=1):
        center = centers[idx]
        if type(name) == tuple and (("DRO" in name[0]) or ("DRO" in name[1])):
            curr_color = "C1"
            label = "DRO"
        else:
            curr_color = color
            label = "Ours"
        ax.vlines(
            center,
            ymin=np.median(bounds["min"]),
            ymax=np.median(bounds["max"]),
            color=curr_color,
            linewidth=2,
        )

        for limit in ["max", "min"]:
            quantiles = np.quantile(bounds[limit], [0.05, 0.95])

            ax.fill_between(
                [center - 0.1, center + 0.1],
                [quantiles[0]] * 2,
                [quantiles[1]] * 2,
                color=curr_color,
                alpha=0.1,
                hatch="//",
            )

            ax.plot(
                [center - 0.05, center + 0.05],
                [np.median(bounds[limit])] * 2,
                color=curr_color,
                linewidth=2,
                label=label,
            )


def plot_references(
    ax, plotting_df: pd.DataFrame, true_field: str, empirical_field: str
):
    true_coef = plotting_df[true_field].mean()
    min_empirical_coef = plotting_df[empirical_field].min()
    max_empirical_coef = plotting_df[empirical_field].max()
    ax.axhline(
        y=true_coef,
        color="blue",
        linestyle="dashed",
        label="True value",
        linewidth=2,
    )

    min_x, max_x = ax.get_xlim()
    ax.fill_between(
        [min_x - 0.2, max_x + 0.2],
        [min_empirical_coef] * 2,
        [max_empirical_coef] * 2,
        color="olive",
        alpha=0.2,
        label="Naive estimator",
        hatch="//",
    )

    ax.set_xlim(min_x - 0.2, max_x + 0.2)
