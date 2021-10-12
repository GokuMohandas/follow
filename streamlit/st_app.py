# streamlit/st_app.py
# Streamlit application.

import copy
import itertools
import numbers
from argparse import Namespace
from collections import Counter, OrderedDict
from distutils.util import strtobool
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

import streamlit as st
from config import config
from tagifai import data, eval, main, utils


@st.cache()
def load_data():
    # Filepaths
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    features_fp = Path(config.DATA_DIR, "features.json")

    # Load data
    projects = utils.load_dict(filepath=projects_fp)
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    features = utils.load_dict(filepath=features_fp)

    return projects, tags_dict, features


@st.cache
def get_artifacts(run_id: int):
    artifacts = main.load_artifacts(run_id=run_id)
    return artifacts


@st.cache(allow_output_mutation=True)
def get_performance():
    performance = utils.load_dict(filepath=Path(config.CONFIG_DIR, "performance.json"))
    return performance


def get_dict_diffs(d_a: Dict, d_b: Dict, d_a_name="a", d_b_name="b", tolerance=0) -> Dict:
    """Differences between two dictionaries with numerical values.
    Args:
        d_a (Dict): Dictionary with data.
        d_b (Dict): Dictionary to compare to.
        d_a_name (str): Name of dict a.
        d_b_name (str): Name of dict b.
    Returns:
        Dict: Differences between keys with numerical values.
    """
    # Recursively flatten
    d_a = pd.json_normalize(d_a, sep=".").to_dict(orient="records")[0]
    d_b = pd.json_normalize(d_b, sep=".").to_dict(orient="records")[0]
    if d_a.keys() != d_b.keys():
        raise Exception("Cannot compare these dictionaries because they have different keys.")

    # Compare
    diffs = {}
    for key in d_a:
        if isinstance(d_a[key], numbers.Number) and isinstance(d_b[key], numbers.Number):
            diff = d_b[key] - d_a[key]
            if abs(diff) > tolerance:
                diffs[key] = {d_a_name: d_a[key], d_b_name: d_b[key], "diff": diff}

    return diffs


@st.cache
def evaluate_df(df, tags_dict, artifacts):
    # Retrieve artifacts
    params = artifacts["params"]

    # Prepare
    df, tags_above_freq, tags_below_freq = data.prepare(
        df=df,
        include=list(tags_dict.keys()),
        exclude=config.EXCLUDED_TAGS,
        min_tag_freq=int(params.min_tag_freq),
    )

    # Preprocess
    df.text = df.text.apply(
        data.preprocess,
        lower=bool(strtobool(str(params.lower))),
        stem=bool(strtobool(str(params.stem))),
    )

    # Evaluate
    y_true, y_pred, performance = eval.evaluate(df=df, artifacts=artifacts)
    sorted_tags = list(
        OrderedDict(
            sorted(performance["class"].items(), key=lambda tag: tag[1]["f1"], reverse=True)
        ).keys()
    )

    return y_true, y_pred, performance, sorted_tags, df


# Title
st.title("Tagifai · MLOps · Made With ML")
"""by [Goku Mohandas](https://twitter.com/GokuMohandas)"""
st.info("🔍 Explore the different pages below.")

# Pages
pages = ["Data", "Performance", "Inference", "Inspection"]
st.header("Pages")
selected_page = st.radio("Select a page:", pages, index=0)

if selected_page == "Data":
    st.header("Data")

    # Load data
    projects, tags_dict, features = load_data()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Projects (sample)")
        st.write(projects[0])
    with col2:
        st.subheader("Tags")
        tag = st.selectbox("Choose a tag", list(tags_dict.keys()))
        st.write(tags_dict[tag])

    # Dataframe
    df = pd.DataFrame(features)
    st.text(f"Projects (count: {len(df)}):")
    st.write(df)

    # Filter tags
    st.write("---")
    st.subheader("Labeling")
    st.write(
        "We want to determine what the minimum tag frequency is so that we have enough samples per tag for training."
    )
    min_tag_freq = st.slider("min_tag_freq", min_value=1, value=30, step=1)
    df, tags_above_freq, tags_below_freq = data.prepare(
        df=df,
        include=list(tags_dict.keys()),
        exclude=config.EXCLUDED_TAGS,
        min_tag_freq=min_tag_freq,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Most common tags**:")
        for item in tags_above_freq.most_common(5):
            st.write(item)
    with col2:
        st.write("**Tags that just made the cut**:")
        for item in tags_above_freq.most_common()[-5:]:
            st.write(item)
    with col3:
        st.write("**Tags that just missed the cut**:")
        for item in tags_below_freq.most_common(5):
            st.write(item)
    with st.expander("Excluded tags"):
        st.write(config.EXCLUDED_TAGS)

    # Number of tags per project
    st.write("---")
    st.subheader("Exploratory Data Analysis")
    num_tags_per_project = [len(tags) for tags in df.tags]
    num_tags, num_projects = zip(*Counter(num_tags_per_project).items())
    plt.figure(figsize=(10, 3))
    ax = sns.barplot(list(num_tags), list(num_projects))
    plt.title("Tags per project", fontsize=20)
    plt.xlabel("Number of tags", fontsize=16)
    ax.set_xticklabels(range(1, len(num_tags) + 1), rotation=0, fontsize=16)
    plt.ylabel("Number of projects", fontsize=16)
    plt.show()
    st.pyplot(plt)

    # Distribution of tags
    tags = list(itertools.chain.from_iterable(df.tags.values))
    tags, tag_counts = zip(*Counter(tags).most_common())
    plt.figure(figsize=(10, 3))
    ax = sns.barplot(list(tags), list(tag_counts))
    plt.title("Tag distribution", fontsize=20)
    plt.xlabel("Tag", fontsize=16)
    ax.set_xticklabels(tags, rotation=90, fontsize=14)
    plt.ylabel("Number of projects", fontsize=16)
    plt.show()
    st.pyplot(plt)

    # Plot word clouds top top tags
    plt.figure(figsize=(20, 8))
    tag = st.selectbox("Choose a tag", tags, index=0)
    subset = df[df.tags.apply(lambda tags: tag in tags)]
    text = subset.text.values
    cloud = WordCloud(
        stopwords=STOPWORDS,
        background_color="black",
        collocations=False,
        width=500,
        height=300,
    ).generate(" ".join(text))
    plt.axis("off")
    plt.imshow(cloud)
    st.pyplot(plt)

    # Preprocessing
    st.write("---")
    st.subheader("Preprocessing")
    text = st.text_input("Input text", "Conditional generation using Variational Autoencoders.")
    filters = st.text_input("filters", "[!\"'#$%&()*+,-./:;<=>?@\\[]^_`{|}~]")
    lower = st.checkbox("lower", True)
    stem = st.checkbox("stem", False)
    preprocessed_text = data.preprocess(text=text, lower=lower, stem=stem)
    st.text("Preprocessed text:")
    st.write(preprocessed_text)

elif selected_page == "Performance":
    st.header("Performance")
    performance_prod = get_performance()
    performance_local = copy.deepcopy(performance_prod)

    # Inject changes
    performance_local["overall"]["precision"] = performance_local["overall"]["precision"] + 0.03
    performance_local["overall"]["recall"] = performance_local["overall"]["recall"] - 0.03
    performance_local["slices"]["overall"]["f1"] = (
        performance_local["slices"]["overall"]["f1"] + 0.05
    )

    # Diffs
    st.write("Differences")
    diffs = get_dict_diffs(
        d_a=performance_prod, d_b=performance_local, d_a_name="prod", d_b_name="local"
    )
    cols = st.columns(3)
    for i, col in enumerate(cols):
        metric = list(diffs.keys())[i]
        col.metric(metric, f'{diffs[metric]["local"]:.2f}', f'{diffs[metric]["diff"]:.2f}')

    # Full
    col1, col2 = st.columns(2)
    with col1:
        st.write("Production")
        st.json(performance_prod)
    with col2:
        st.write("Local")
        st.json(performance_local)

    st.warning(
        "We're simulating previous performance but we could easily pull \
                directly from the repository history, model store, etc. \
                We could also graph improvements/regression over multiple \
                releases, show impact of data over time, etc."
    )

elif selected_page == "Inference":
    st.header("Inference")
    text = st.text_input(
        "Enter text:",
        "Transfer learning with transformers for self-supervised learning.",
    )
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    prediction = main.predict_tags(text=text, run_id=run_id)
    st.text("Prediction:")
    st.write(prediction)

elif selected_page == "Inspection":
    st.header("Inspection")
    st.write(
        "We're going to inspect the TP, FP and FN samples across our different tags. It's a great way to catch issues with labeling (FP), weaknesses (FN), etc."
    )

    # Load and process data
    params = Namespace(**utils.load_dict(filepath=Path(config.CONFIG_DIR, "params.json")))
    projects, tags_dict, features = load_data()
    df = pd.DataFrame(features)
    df, tags_above_freq, tags_below_freq = data.prepare(
        df=df,
        include=list(tags_dict.keys()),
        exclude=config.EXCLUDED_TAGS,
        min_tag_freq=params.min_tag_freq,
    )
    df.text = df.text.apply(data.preprocess, lower=params.lower, stem=params.stem)

    # Get performance
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = get_artifacts(run_id=run_id)
    label_encoder = artifacts["label_encoder"]
    y_true, y_pred, performance, sorted_tags, df = evaluate_df(
        df=df,
        tags_dict=tags_dict,
        artifacts=artifacts,
    )
    tag = st.selectbox("Choose a tag", sorted_tags, index=0)
    st.json(performance["class"][tag])

    # TP, FP, FN samples
    index = label_encoder.class_to_index[tag]
    tp, fp, fn = [], [], []
    for i in range(len(y_true)):
        true = y_true[i][index]
        pred = y_pred[i][index]
        if true and pred:
            tp.append(i)
        elif not true and pred:
            fp.append(i)
        elif true and not pred:
            fn.append(i)

    # Samples
    num_samples = 3
    cm = [(tp, "True positives"), (fp, "False positives"), (fn, "False negatives")]
    for item in cm:
        if len(item[0]):
            with st.expander(item[1]):
                for i in item[0][:num_samples]:
                    st.write(f"**[id: {df.id.iloc[i]}]** {df.text.iloc[i]}")
                    st.text("True")
                    st.write(label_encoder.decode([y_true[i]])[0])
                    st.text("Predicted")
                    st.write(label_encoder.decode([y_pred[i]])[0])
    st.write("\n")
    st.warning(
        "Be careful not to make decisions based on predicted probabilities before [calibrating](https://arxiv.org/abs/1706.04599) them to reliably use as measures of confidence."
    )
    """
    ### Extensions

    - Use false positives to identify potentially mislabeled data.
    - Connect inspection pipelines with annotation systems so that changes to the data can be reviewed and incorporated.
    - Inspect FP / FN samples by [estimating training data influences (TracIn)](https://arxiv.org/abs/2002.08484) on their predictions.
    - Inspect the trained model's behavior under various conditions using the [WhatIf](https://pair-code.github.io/what-if-tool/) tool.
    """


else:
    st.text("Please select a valid page option from above...")

st.write("---")

# Resources
"""
## Resources

- 🎓 Lessons: https://madewithml.com/
- 🐙 Repository: https://github.com/GokuMohandas/MLOps
- 📘 Documentation: https://gokumohandas.github.io/MLOps
- 📬 Subscribe: https://newsletter.madewithml.com
"""
