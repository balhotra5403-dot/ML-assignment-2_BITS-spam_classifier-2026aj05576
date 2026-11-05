 =============================================================================

#  app.py  -  Streamlit web application

# -----------------------------------------------------------------------------

#  Machine Learning Assignment 2  -  Spam E-mail Classification (UCI Spambase)

#

#  WHAT THIS APP DOES:

#    This is an interactive web page (built with Streamlit) that lets a user:

#       (a) upload a TEST data CSV file,

#       (b) pick one of the five trained models from a dropdown,

#       (c) see the six evaluation metrics for that model, and

#       (d) see a confusion matrix and a classification report.

#

#  IMPORTANT (assignment rule): the app only LOADS the already-trained models

#  from the  model/  folder and evaluates them on the uploaded test CSV. It does

#  NOT re-train anything. Training happens separately in  train_models.py .

#

#  HOW TO RUN LOCALLY:

#      streamlit run app.py

#

#  HOW IT RUNS ON STREAMLIT COMMUNITY CLOUD:

#      The cloud automatically runs "streamlit run app.py" for you after it

#      installs the libraries listed in requirements.txt.

# =============================================================================

 

# ----- Standard library ------------------------------------------------------

import json                         # read the JSON metadata saved by training

import os                           # build file paths

 

# ----- Third-party libraries -------------------------------------------------

import numpy as np

import pandas as pd

import streamlit as st              # the web-app framework

import matplotlib.pyplot as plt     # for drawing the confusion matrix

import seaborn as sns               # nicer heatmap for the confusion matrix

import joblib                       # load the saved .joblib models

 

# The same six metrics used during training, plus helpers for the display.

from sklearn.metrics import (

    accuracy_score,

    roc_auc_score,

    precision_score,

    recall_score,

    f1_score,

    matthews_corrcoef,

    confusion_matrix,

    classification_report,

)

 

# =============================================================================

#  CONFIGURATION

# =============================================================================

 

# Directory of this file, so the app finds the model/ folder no matter where it

# is launched from (local machine or Streamlit Cloud).

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(HERE, "model")

 

# Name of the label column that must be present in the uploaded CSV.

TARGET_COLUMN = "target"

 

# Human-readable labels for the two classes (0 and 1).

CLASS_LABELS = ["Not Spam (0)", "Spam (1)"]

 

# =============================================================================

#  CACHED LOADERS

# -----------------------------------------------------------------------------

#  @st.cache_resource tells Streamlit to load these files only ONCE and reuse

#  them across reruns, which keeps the app fast. (Streamlit re-runs the whole

#  script every time the user interacts with a widget.)

# =============================================================================

 

@st.cache_resource

def load_metadata():

    """Load the three small JSON files created by train_models.py."""

    with open(os.path.join(MODEL_DIR, "feature_names.json")) as fh:

        feature_names = json.load(fh)

    with open(os.path.join(MODEL_DIR, "model_files.json")) as fh:

        model_files = json.load(fh)        # {display name: file name}

    with open(os.path.join(MODEL_DIR, "metrics.json")) as fh:

        reference_metrics = json.load(fh)  # metrics measured during training

    return feature_names, model_files, reference_metrics

 

@st.cache_resource

def load_model(file_name):

    """Load one trained pipeline (scaler + classifier) from the model/ folder."""

    return joblib.load(os.path.join(MODEL_DIR, file_name))

 

# =============================================================================

#  HELPER FUNCTIONS

# =============================================================================

 

def compute_metrics(model, X, y_true):

    """Return the six evaluation metrics for one model on one dataset."""

    y_pred = model.predict(X)                      # hard 0/1 predictions

    y_score = model.predict_proba(X)[:, 1]         # probability of spam (class 1)

    return {

        "Accuracy": accuracy_score(y_true, y_pred),

        "AUC": roc_auc_score(y_true, y_score),

        "Precision": precision_score(y_true, y_pred),

        "Recall": recall_score(y_true, y_pred),

        "F1": f1_score(y_true, y_pred),

        "MCC": matthews_corrcoef(y_true, y_pred),

    }, y_pred

 

def validate_and_split(df, feature_names):

    """Check the uploaded CSV and split it into features X and label y.

 

    Returns (X, y, error_message). If error_message is not None, something is

    wrong with the uploaded file and X / y should be ignored.

    """

    # 1) The label column must be present.

    if TARGET_COLUMN not in df.columns:

        return None, None, (

            "The uploaded CSV must contain a '%s' column (the true label). "

            "Please upload the provided test_data.csv." % TARGET_COLUMN

        )

 

    # 2) Every feature the models were trained on must be present.

    missing = [c for c in feature_names if c not in df.columns]

    if missing:

        return None, None, (

            "The uploaded CSV is missing %d required feature column(s), "

            "for example: %s" % (len(missing), ", ".join(missing[:5]))

        )

 

    # 3) Select the feature columns in the EXACT order used during training.

    #    (Re-ordering protects against a CSV whose columns are shuffled.)

    X = df[feature_names].copy()

    y = df[TARGET_COLUMN].astype(int).copy()

 

    # 4) The label must have both classes for AUC to be defined.

    if y.nunique() < 2:

        return None, None, (

            "The 'target' column must contain BOTH classes (0 and 1) so that "

            "metrics like AUC can be computed."

        )

 

    return X, y, None

 

def plot_confusion_matrix(y_true, y_pred):

    """Draw a confusion matrix as a labelled heatmap and return the figure."""

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    sns.heatmap(

        cm,

        annot=True,              # write the count inside each cell

        fmt="d",                 # integer format

        cmap="Blues",

        cbar=False,

        xticklabels=CLASS_LABELS,

        yticklabels=CLASS_LABELS,

        ax=ax,

    )

    ax.set_xlabel("Predicted label")

    ax.set_ylabel("True label")

    ax.set_title("Confusion Matrix")

    fig.tight_layout()

    return fig

 

# =============================================================================

#  EXTRA HELPERS FOR THE ENHANCED UI

# =============================================================================

 

@st.cache_data

def load_sample_data():

    """Load the bundled test_data.csv so users can preview/download a sample."""

    path = os.path.join(HERE, "test_data.csv")

    if os.path.exists(path):

        return pd.read_csv(path)

    return None

 

def plot_metrics_comparison(results_df):

    """Grouped bar chart comparing all six metrics across every model."""

    metrics_order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    fig, ax = plt.subplots(figsize=(10, 4.6))

    n_models = len(results_df)

    n_metrics = len(metrics_order)

    x = np.arange(n_models)          # one group of bars per model

    width = 0.13                     # width of each individual bar

    for i, metric in enumerate(metrics_order):

        offset = (i - n_metrics / 2) * width + width / 2

        ax.bar(x + offset, results_df[metric].values, width, label=metric)

    ax.set_xticks(x)

    ax.set_xticklabels(results_df.index, rotation=20, ha="right")

    ax.set_ylim(0, 1.08)

    ax.set_ylabel("Score")

    ax.set_title("Evaluation metrics by model")

    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()

    return fig

 

# =============================================================================

#  PAGE LAYOUT

# =============================================================================

 

# Set the browser tab title/icon and use a wide layout.

st.set_page_config(

    page_title="Spam Classifier - ML Assignment 2",

    page_icon="\U0001F4E7",          # envelope icon in the browser tab

    layout="wide",

)

 

st.title("\U0001F4E7 Spam E-mail Classification - Model Evaluation App")

st.write(

    "Upload a **test CSV**, choose a trained model, and view its evaluation "

    "metrics, confusion matrix and classification report. "

    "The models are pre-trained on the UCI Spambase dataset and are only "

    "**loaded** here - they are never re-trained on your uploaded data."

)

 

# Load the metadata once (feature names, model file map, reference metrics).

feature_names, model_files, reference_metrics = load_metadata()

sample_df = load_sample_data()       # bundled test data for preview/download

 

# -----------------------------------------------------------------------------

#  SIDEBAR  -  the user's controls live here

# -----------------------------------------------------------------------------

st.sidebar.header("1. Upload test data")

uploaded_file = st.sidebar.file_uploader(

    "Upload your test CSV (must include the 'target' column)",

    type=["csv"],

)

 

st.sidebar.header("2. Choose a model")

# FEATURE (b): model selection dropdown. "Compare all models" is an extra option.

model_choice = st.sidebar.selectbox(

    "Select a classification model",

    options=["Compare all models"] + list(model_files.keys()),

)

 

# ---- Download sample data so a first-time user has something to upload -------

st.sidebar.header("3. Need a sample file?")

if sample_df is not None:

    st.sidebar.download_button(

        "\u2B07\uFE0F Download sample test data (25 rows)",

        data=sample_df.head(25).to_csv(index=False).encode("utf-8"),

        file_name="sample_test_data.csv",

        mime="text/csv",

        use_container_width=True,

    )

    st.sidebar.download_button(

        "\u2B07\uFE0F Download full test data (%d rows)" % len(sample_df),

        data=sample_df.to_csv(index=False).encode("utf-8"),

        file_name="test_data.csv",

        mime="text/csv",

        use_container_width=True,

    )

st.sidebar.caption(

    "Tip: download a sample above, then upload it to see the app in action."

)

 

# -----------------------------------------------------------------------------

#  MAIN PAGE  -  single scrollable page:

#     Section 1 (top):    Model evaluation + graphs

#     Section 2 (below):  Dataset description & sample

# -----------------------------------------------------------------------------

 

# ======================= SECTION 1: MODEL EVALUATION ========================

st.header("\U0001F916 Model Evaluation")

 

if uploaded_file is None:

    st.info(

        "Please upload a test CSV from the sidebar (or download a sample "

        "first). Meanwhile, here are the reference metrics from training."

    )

    ref_df = pd.DataFrame(reference_metrics).T.round(4)

    st.subheader("Reference metrics (hold-out test set, during training)")

    st.dataframe(ref_df, use_container_width=True)

    # A graph of the reference metrics so the page is informative even before upload.

    st.subheader("Reference metrics - visual comparison")

    st.pyplot(plot_metrics_comparison(ref_df))

else:

    # Read the uploaded CSV into a DataFrame.

    data = None

    try:

        data = pd.read_csv(uploaded_file)

    except Exception as exc:

        st.error("Could not read the uploaded file as CSV: %s" % exc)

 

    if data is not None:

        st.subheader("Preview of uploaded data")

        st.write("Shape: %d rows x %d columns" % (data.shape[0], data.shape[1]))

        st.dataframe(data.head(), use_container_width=True)

 

        # Validate the file and split into features / label.

        X, y, error = validate_and_split(data, feature_names)

        if error is not None:

            st.error(error)

        else:

            st.success("File looks good: %d rows ready for evaluation." % len(X))

 

            # ---------------------------------------------------------------

            #  CASE 1: Compare ALL models side by side

            # ---------------------------------------------------------------

            if model_choice == "Compare all models":

                st.subheader("Evaluation metrics for all models")

 

                rows = []

                for display_name, file_name in model_files.items():

                    model = load_model(file_name)

                    metrics, _ = compute_metrics(model, X, y)

                    metrics_row = {"Model": display_name}

                    metrics_row.update({k: round(v, 4) for k, v in metrics.items()})

                    rows.append(metrics_row)

 

                results_df = pd.DataFrame(rows).set_index("Model")

                # FEATURE (c): display of evaluation metrics (every model).

                st.dataframe(results_df, use_container_width=True)

 

                best_model = results_df["MCC"].idxmax()

                st.success("Best model on this data (by MCC): **%s**" % best_model)

 

                # Graph: all six metrics compared across the five models.

                st.subheader("Metrics comparison graph")

                st.pyplot(plot_metrics_comparison(results_df))

 

            # ---------------------------------------------------------------

            #  CASE 2: Evaluate a SINGLE selected model

            # ---------------------------------------------------------------

            else:

                file_name = model_files[model_choice]

                model = load_model(file_name)

                metrics, y_pred = compute_metrics(model, X, y)

 

                st.subheader("Evaluation metrics - %s" % model_choice)

 

                # FEATURE (c): six metric cards.

                c1, c2, c3 = st.columns(3)

                c1.metric("Accuracy", "%.4f" % metrics["Accuracy"])

                c2.metric("AUC", "%.4f" % metrics["AUC"])

                c3.metric("Precision", "%.4f" % metrics["Precision"])

                c4, c5, c6 = st.columns(3)

                c4.metric("Recall", "%.4f" % metrics["Recall"])

                c5.metric("F1 Score", "%.4f" % metrics["F1"])

                c6.metric("MCC", "%.4f" % metrics["MCC"])

 

                # Graph: this model's six metrics as a bar chart.

                st.subheader("Metrics graph - %s" % model_choice)

                metric_series = pd.Series(

                    {k: metrics[k] for k in

                     ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}

                )

                st.bar_chart(metric_series)

 

                # FEATURE (d): confusion matrix AND classification report.

                st.subheader("Confusion matrix and classification report")

                left, right = st.columns(2)

                with left:

                    fig = plot_confusion_matrix(y, y_pred)

                    st.pyplot(fig)

                with right:

                    report = classification_report(

                        y, y_pred,

                        target_names=CLASS_LABELS,

                        output_dict=True,

                        zero_division=0,

                    )

                    report_df = pd.DataFrame(report).T.round(4)

                    st.dataframe(report_df, use_container_width=True)

 

st.divider()

 

# ======================= SECTION 2: DATASET & SAMPLE ========================

st.header("\U0001F4CA Dataset & Sample")

st.subheader("About the dataset - UCI Spambase")

 

# Headline numbers shown as metric cards.

d1, d2, d3, d4 = st.columns(4)

d1.metric("Instances (rows)", "4,601")

d2.metric("Features", "57")

d3.metric("Classes", "2")

d4.metric("Test rows provided", "921")

 

st.write(

    "Each row describes one e-mail using **57 numeric features**. The task "

    "is to predict whether the e-mail is **spam (1)** or **not-spam (0)** - "

    "a binary classification problem. The 57 features fall into three groups:"

)

 

# Feature-group breakdown table.

groups_df = pd.DataFrame(

    {

        "Feature group": [

            "word_freq_* (word frequencies)",

            "char_freq_* (character frequencies)",

            "capital_run_length_* (capital-letter runs)",

            "Total features",

        ],

        "Count": [48, 6, 3, 57],

        "Meaning": [

            "% of words in the e-mail matching a given word (e.g. free, money)",

            "% of characters matching a symbol ( ; ( [ ! $ # )",

            "average / longest / total length of capital-letter runs",

            "all input features combined",

        ],

    }

)

st.dataframe(groups_df, use_container_width=True, hide_index=True)

 

st.divider()

 

# ---- What a single data point looks like ------------------------------------

st.subheader("What a single data point looks like")

st.write(

    "A data point is one e-mail: 57 feature values plus its true label. "

    "Pick any row below to inspect all its feature values."

)

if sample_df is not None:

    row_idx = st.number_input(

        "Row number to inspect",

        min_value=0, max_value=len(sample_df) - 1, value=0, step=1,

    )

    one_row = sample_df.iloc[int(row_idx)]

    true_label = int(one_row[TARGET_COLUMN])

    st.info("True label of this e-mail: **%s**" % CLASS_LABELS[true_label])

    # Show the single data point as a readable "feature -> value" table.

    point_df = one_row.drop(labels=[TARGET_COLUMN]).to_frame(name="value")

    point_df.index.name = "feature"

    st.dataframe(point_df, use_container_width=True, height=380)

 

    st.subheader("Sample rows from the test data")

    st.dataframe(sample_df.head(10), use_container_width=True)

else:

    st.warning("Sample data file (test_data.csv) was not found in the app.")

 

# -----------------------------------------------------------------------------

#  Footer

# -----------------------------------------------------------------------------

st.caption(

    "ML Assignment 2 - UCI Spambase - 5 classifiers "

    "(Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM)."

)
