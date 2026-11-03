# =============================================================================

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

#  PAGE LAYOUT

# =============================================================================

 

# Set the browser tab title and use a wide layout.

st.set_page_config(page_title="Spam Classifier - ML Assignment 2", layout="wide")

 

st.title("Spam E-mail Classification - Model Evaluation App")

st.write(

    "Upload a **test CSV**, choose a trained model, and view its evaluation "

    "metrics, confusion matrix and classification report. "

    "The models are pre-trained on the UCI Spambase dataset and are only "

    "**loaded** here - they are never re-trained on your uploaded data."

)

 

# Load the metadata once (feature names, model file map, reference metrics).

feature_names, model_files, reference_metrics = load_metadata()

 

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

 

st.sidebar.caption(

    "Tip: use the provided **test_data.csv** from the repository. It already "

    "contains all required feature columns plus the 'target' label."

)

 

# -----------------------------------------------------------------------------

#  MAIN AREA  -  reacts to what the user uploaded / selected

# -----------------------------------------------------------------------------

 

# If no file has been uploaded yet, show a friendly instruction and stop.

if uploaded_file is None:

    st.info("Please upload a test CSV file from the sidebar to begin.")

    # Show the reference metrics (measured during training) so the page is not

    # empty - this is purely informational.

    st.subheader("Reference metrics (measured on the hold-out test set during training)")

    ref_df = pd.DataFrame(reference_metrics).T.round(4)

    st.dataframe(ref_df, use_container_width=True)

    st.stop()   # nothing more to do until a file is uploaded

 

# Read the uploaded CSV into a DataFrame.

try:

    data = pd.read_csv(uploaded_file)

except Exception as exc:

    st.error("Could not read the uploaded file as CSV: %s" % exc)

    st.stop()

 

st.subheader("Preview of uploaded data")

st.write("Shape: %d rows x %d columns" % (data.shape[0], data.shape[1]))

st.dataframe(data.head(), use_container_width=True)

 

# Validate the file and split into features / label.

X, y, error = validate_and_split(data, feature_names)

if error is not None:

    st.error(error)

    st.stop()

 

st.success("File looks good: %d rows ready for evaluation." % len(X))

 

# -----------------------------------------------------------------------------

#  CASE 1: Compare ALL models side by side

# -----------------------------------------------------------------------------

if model_choice == "Compare all models":

    st.subheader("Evaluation metrics for all models (on your uploaded test data)")

 

    rows = []

    for display_name, file_name in model_files.items():

        model = load_model(file_name)

        metrics, _ = compute_metrics(model, X, y)

        metrics_row = {"Model": display_name}

        metrics_row.update({k: round(v, 4) for k, v in metrics.items()})

        rows.append(metrics_row)

 

    results_df = pd.DataFrame(rows).set_index("Model")

    # FEATURE (c): display of evaluation metrics (for every model here).

    st.dataframe(results_df, use_container_width=True)

 

    # Highlight the best model by MCC (a balanced single-number score).

    best_model = results_df["MCC"].idxmax()

    st.success("Best model on this data (by MCC): **%s**" % best_model)

 

    # A simple bar chart comparing accuracy across models.

    st.bar_chart(results_df["Accuracy"])

 

# -----------------------------------------------------------------------------

#  CASE 2: Evaluate a SINGLE selected model

# -----------------------------------------------------------------------------

else:

    file_name = model_files[model_choice]

    model = load_model(file_name)

    metrics, y_pred = compute_metrics(model, X, y)

 

    st.subheader("Evaluation metrics - %s" % model_choice)

 

    # FEATURE (c): display of evaluation metrics, shown as six metric cards.

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", "%.4f" % metrics["Accuracy"])

    c2.metric("AUC", "%.4f" % metrics["AUC"])

    c3.metric("Precision", "%.4f" % metrics["Precision"])

    c4, c5, c6 = st.columns(3)

    c4.metric("Recall", "%.4f" % metrics["Recall"])

    c5.metric("F1 Score", "%.4f" % metrics["F1"])

    c6.metric("MCC", "%.4f" % metrics["MCC"])

 

    # FEATURE (d): confusion matrix AND classification report, side by side.

    st.subheader("Confusion matrix and classification report")

    left, right = st.columns(2)

 

    with left:

        fig = plot_confusion_matrix(y, y_pred)

        st.pyplot(fig)

 

    with right:

        # classification_report returns a dict we turn into a tidy table.

        report = classification_report(

            y, y_pred,

            target_names=CLASS_LABELS,

            output_dict=True,

            zero_division=0,

        )

        report_df = pd.DataFrame(report).T.round(4)

        st.dataframe(report_df, use_container_width=True)

 

# -----------------------------------------------------------------------------

#  Footer

# -----------------------------------------------------------------------------

st.caption(

    "ML Assignment 2 - UCI Spambase - 5 classifiers "

    "(Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM)."

)
