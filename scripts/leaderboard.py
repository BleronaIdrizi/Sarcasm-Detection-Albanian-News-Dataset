from html import escape
from inspect import currentframe

import pandas as pd
from IPython.display import HTML, display


def _get_model_comparisons_from_caller():
    frame = currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return None

    return frame.f_back.f_back.f_globals.get("MODEL_COMPARISONS")


def show_leaderboard(model_comparisons=None):
    if model_comparisons is None:
        model_comparisons = _get_model_comparisons_from_caller()

    if not model_comparisons:
        print("No models evaluated yet.")
        return

    dataframe = pd.DataFrame(model_comparisons).copy()
    metric_cols = [
        "accuracy_mean", "accuracy_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std",
    ]
    display_cols = ["dataset", "model"] + metric_cols

    dataset_order = list(dict.fromkeys(dataframe["dataset"].tolist()))
    dataframe["_dataset_order"] = pd.Categorical(
        dataframe["dataset"],
        categories=dataset_order,
        ordered=True,
    )
    dataframe = (
        dataframe
        .sort_values(["_dataset_order", "f1_mean"], ascending=[True, False])
        .drop(columns="_dataset_order")
        .reset_index(drop=True)
    )

    html_rows = [
        "<style>",
        ".leaderboard-table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 13px; color: #f4f4f5; background: #181818; border: 1px solid #3a3a3a; border-radius: 8px; overflow: hidden; }",
        ".leaderboard-table th, .leaderboard-table td { border-bottom: 1px solid #343434; padding: 9px 11px; }",
        ".leaderboard-table th { background: #242424; color: #f7f7f7; text-align: center; font-weight: 700; }",
        ".leaderboard-table tbody tr:nth-child(even) td:not(.dataset-cell) { background: #1f1f1f; }",
        ".leaderboard-table tbody tr:nth-child(odd) td:not(.dataset-cell) { background: #191919; }",
        ".leaderboard-table td { text-align: right; color: #eeeeee; }",
        ".leaderboard-table td.model-cell { text-align: left; font-weight: 600; color: #f5f5f5; }",
        ".leaderboard-table td.dataset-cell { text-align: center; vertical-align: middle; font-weight: 700; color: #ecfeff; background: #164e50; min-width: 190px; border-right: 1px solid #3a3a3a; }",
        ".leaderboard-table tr.group-start td { box-shadow: inset 0 3px 0 #164e50; }",
        "</style>",
        '<table class="leaderboard-table">',
        "  <thead>",
        "    <tr>",
    ]

    for col in display_cols:
        html_rows.append(f"      <th>{escape(col)}</th>")

    html_rows.extend(["    </tr>", "  </thead>", "  <tbody>"])

    for group_idx, (dataset, group) in enumerate(dataframe.groupby("dataset", sort=False)):
        group = group.reset_index(drop=True)
        rowspan = len(group)
        for idx, row in group.iterrows():
            row_class = ' class="group-start"' if group_idx > 0 and idx == 0 else ""
            html_rows.append(f"    <tr{row_class}>")
            if idx == 0:
                html_rows.append(
                    f'      <td class="dataset-cell" rowspan="{rowspan}">{escape(str(dataset))}</td>'
                )
            html_rows.append(f'      <td class="model-cell">{escape(str(row["model"]))}</td>')
            for col in metric_cols:
                html_rows.append(f"      <td>{row[col]:.4f}</td>")
            html_rows.append("    </tr>")

    html_rows.extend(["  </tbody>", "</table>"])
    display(HTML("\n".join(html_rows)))
