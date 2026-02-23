import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Noto Sans CJK SC", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def build_data_profile(df):
    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_by_column": {column: int(value) for column, value in df.isnull().sum().items()},
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }
    return profile


def EDA(df, output_path="output/analysis/data_profile.json"):
    profile = build_data_profile(df)

    print("=" * 100)
    print(f"===> This data frame contains {profile['rows']} rows and {profile['columns']} columns")
    print("=" * 100)

    print(
        "{:20}{:18}{:18}{}".format(
            "FEATURE NAME",
            "DATA FORMAT",
            "MISSING VALUES",
            "FIRST FEW SAMPLES",
        )
    )

    missing_sorted = df.isnull().sum().sort_values(ascending=False)
    for feature_name in missing_sorted.index:
        dtype_value = str(df[feature_name].dtype)
        missing_value = int(missing_sorted[feature_name])
        samples = ",".join(str(value) for value in df[feature_name].head(5).tolist())
        print("{:20}{:18}{:18}{}".format(feature_name, dtype_value, missing_value, samples))

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(profile, file, ensure_ascii=False, indent=2)

    return df


def visulaisation(df, output_dir="output/visualisation", show=False):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pie_chart(
        df,
        "TRADE_ZONE",
        "STORE_CODE",
        "TRADE ZONE Distribution",
        output_path / "TRADE ZONE DISTRIBUTION.png",
        show=show,
    )
    pie_chart(
        df,
        "LEVEL_ID",
        "STORE_CODE",
        "LEVEL ID Distribution",
        output_path / "LEVEL_ID Barchart.png",
        show=show,
    )
    bar_chart(df, output_path / "Review Length Barchart.png", show=show)

    return df


def bar_chart(df, file_path, show=False):
    review_length_bar = df.groupby(["content_len"], as_index=False)["STORE_CODE"].count()
    review_length_bar.columns = ["content_len", "frequency"]
    review_length_bar = review_length_bar.sort_values("content_len")

    plt.figure(figsize=(10, 5), dpi=120)
    sns.lineplot(data=review_length_bar, x="content_len", y="frequency")
    plt.title("Customer review length")
    plt.xlabel("content_len")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(file_path)
    if show:
        plt.show()
    plt.close()


def pie_chart(dataframe, col, target, title, file_path, show=False):
    target_df = dataframe.groupby([col], dropna=False)[target].agg(["count"]).reset_index()
    target_df = target_df.sort_values("count", ascending=False)
    labels = target_df[col].astype(str).tolist()
    colors = sns.color_palette("pastel")[0 : len(labels)]

    plt.figure(figsize=(10, 5), dpi=120)
    plt.pie(
        target_df["count"],
        labels=labels,
        autopct="%1.2f%%",
        startangle=45,
        colors=colors,
        labeldistance=0.75,
        pctdistance=0.4,
    )
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.legend(labels, loc="best")
    plt.tight_layout()
    plt.savefig(file_path)
    if show:
        plt.show()
    plt.close()


def data_process(dataframe, datetimecol, wordcol):
    processed_df = dataframe.copy()
    processed_df[datetimecol] = pd.to_datetime(processed_df[datetimecol], errors="coerce")
    processed_df["Year"] = processed_df[datetimecol].dt.year
    processed_df["month"] = processed_df[datetimecol].dt.month
    processed_df["day"] = processed_df[datetimecol].dt.day
    processed_df["hour"] = processed_df[datetimecol].dt.hour
    processed_df["content_len"] = processed_df[wordcol].fillna("").astype(str).str.len()

    return processed_df
