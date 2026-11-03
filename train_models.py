# =============================================================================

#  train_models.py

# -----------------------------------------------------------------------------

#  Machine Learning Assignment 2  -  Spam E-mail Classification (UCI Spambase)

#

#  WHAT THIS SCRIPT DOES (high-level):

#    1. Loads the UCI "Spambase" dataset (4601 e-mails, 57 numeric features).

#    2. Gives every column a clean, human-readable name.

#    3. Splits the data into a TRAIN set and a TEST set (80% / 20%).

#    4. Trains FIVE classification models, each wrapped in a Pipeline that also

#       standard-scales the features:

#            - Logistic Regression

#            - Decision Tree

#            - K-Nearest Neighbours (KNN)

#            - Gaussian Naive Bayes

#            - Support Vector Machine (SVM)

#    5. Evaluates every model on the TEST set using SIX metrics:

#            Accuracy, AUC, Precision, Recall, F1, Matthews Corr. Coeff. (MCC)

#    6. Saves every trained model to the  model/  folder as a .joblib file.

#    7. Saves the list of feature names and the reference metrics as JSON so the

#       Streamlit app can reuse them.

#    8. Writes the held-out TEST set to  test_data.csv  (this is the file you

#       will later upload inside the Streamlit web app).

#

#  WHY A PIPELINE?  A scikit-learn Pipeline stores the scaler AND the model

#  together in one object. When the app loads the saved pipeline it can call

#  predict() directly on RAW (unscaled) data - the pipeline scales it

#  automatically. This removes a whole class of "I forgot to scale" bugs.

#

#  HOW TO RUN:

#      python train_models.py

# =============================================================================

 

# ----- Standard library imports ----------------------------------------------

import json                         # to save feature names / metrics as .json

import os                           # to build file paths that work on any OS

 

# ----- Third-party (data science) imports ------------------------------------

import numpy as np                  # numerical arrays

import pandas as pd                 # tabular data (DataFrame)

 

# scikit-learn building blocks

from sklearn.model_selection import train_test_split   # split data into train/test

from sklearn.preprocessing import StandardScaler       # feature standardisation

from sklearn.pipeline import Pipeline                  # chain scaler + model

 

# The five classification algorithms required by the assignment

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

 

# The six evaluation metrics required by the assignment

from sklearn.metrics import (

    accuracy_score,

    roc_auc_score,

    precision_score,

    recall_score,

    f1_score,

    matthews_corrcoef,

)

 

import joblib                       # to save/load trained models to disk

 

# =============================================================================

#  CONFIGURATION  (change these in ONE place if ever needed)

# =============================================================================

 

# A fixed random seed makes the whole run reproducible: the same split and the

# same models are produced every time the script is executed. Reproducibility is

# important for grading and for matching the numbers shown in the README.

RANDOM_STATE = 42

 

# Fraction of the data that is held out for testing (20% -> ~920 e-mails).

TEST_SIZE = 0.20

 

# Folder (relative to this script) where the dataset files live and where the

# outputs (models, csv, json) will be written.

HERE = os.path.dirname(os.path.abspath(__file__))   # directory of this file

DATA_FILE = os.path.join(HERE, "spambase.data")     # raw UCI data (no header)

NAMES_FILE = os.path.join(HERE, "spambase.names")   # file that lists the columns

MODEL_DIR = os.path.join(HERE, "model")             # saved models go here

TEST_CSV = os.path.join(HERE, "test_data.csv")      # test data for the app

 

# The name we give to the column the models predict (0 = not spam, 1 = spam).

TARGET_COLUMN = "target"

 

# =============================================================================

#  STEP 1 - Work out the column names

# -----------------------------------------------------------------------------

#  The raw  spambase.data  file has NO header row. The official column names are

#  listed in  spambase.names . Some of those names contain characters that are

#  awkward inside a CSV header (for example  char_freq_;  or  char_freq_$ ), so

#  we replace those symbols with readable words.

# =============================================================================

 

def clean_name(raw_name):

    """Turn an official column name into a CSV/Python-friendly column name.

 

    Example:  'char_freq_;'  ->  'char_freq_semicolon'

    """

    replacements = {

        ";": "semicolon",

        "(": "lparen",

        "[": "lbracket",

        "!": "exclaim",

        "$": "dollar",

        "#": "hash",

    }

    name = raw_name

    for symbol, word in replacements.items():

        name = name.replace(symbol, word)

    return name

 

def load_feature_names():

    """Read the feature names (in order) from spambase.names.

 

    Every relevant line in that file looks like:

        word_freq_make:          continuous.

    We keep only the part before the colon, for the three known prefixes, in the

    exact order they appear - this order must match the columns in spambase.data.

    """

    prefixes = ("word_freq_", "char_freq_", "capital_run_length_")

    names = []

    with open(NAMES_FILE, "r") as fh:

        for line in fh:

            line = line.strip()

            # A real attribute line starts with one of the known prefixes and

            # contains a ':' separating the name from the "continuous." type.

            if line.startswith(prefixes) and ":" in line:

                raw = line.split(":")[0].strip()

                names.append(clean_name(raw))

    return names

 

# =============================================================================

#  STEP 2 - Load the dataset into a pandas DataFrame

# =============================================================================

 

def load_dataset():

    """Load spambase.data and return (X features DataFrame, y target Series)."""

    feature_names = load_feature_names()

 

    # Safety check: the Spambase dataset must have exactly 57 features. If this

    # ever fails, the names file and data file are out of sync.

    assert len(feature_names) == 57, (

        "Expected 57 feature names, found %d" % len(feature_names)

    )

 

    # All 58 column names = 57 features + the target column at the end.

    all_columns = feature_names + [TARGET_COLUMN]

 

    # Read the comma-separated data file, telling pandas there is no header row

    # and supplying our own column names.

    df = pd.read_csv(DATA_FILE, header=None, names=all_columns)

 

    # Separate the input features (X) from the label we want to predict (y).

    X = df[feature_names]            # all columns except target

    y = df[TARGET_COLUMN].astype(int)  # 0 = not spam, 1 = spam

    return X, y, feature_names

 

# =============================================================================

#  STEP 3 - Define the five models (each as a scaler + classifier Pipeline)

# =============================================================================

 

def build_models():

    """Return a dictionary {display_name: untrained sklearn Pipeline}."""

    # NOTE: every pipeline starts with StandardScaler so the features are put on

    # a comparable scale (mean 0, variance 1). This is essential for Logistic

    # Regression, KNN and SVM; it is harmless for Decision Tree and Naive Bayes.

    models = {

        "Logistic Regression": Pipeline([

            ("scaler", StandardScaler()),

            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),

        ]),

        "Decision Tree": Pipeline([

            ("scaler", StandardScaler()),

            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),

        ]),

        "KNN": Pipeline([

            ("scaler", StandardScaler()),

            ("clf", KNeighborsClassifier(n_neighbors=5)),

        ]),

        "Naive Bayes": Pipeline([

            ("scaler", StandardScaler()),

            ("clf", GaussianNB()),

        ]),

        # SVC(probability=True) lets us call predict_proba(), which we need for

        # the AUC metric. It is a little slower to train but fine for this size.

        "Support Vector Machine": Pipeline([

            ("scaler", StandardScaler()),

            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),

        ]),

    }

    return models

 

# =============================================================================

#  STEP 4 - Evaluate a trained model and return the six metrics

# =============================================================================

 

def evaluate(model, X_test, y_test):

    """Compute the six required metrics for one trained model."""

    # Hard class predictions (0 or 1) used by most metrics.

    y_pred = model.predict(X_test)

 

    # Predicted probability of the positive class (spam = 1). roc_auc_score needs

    # a probability/score rather than a hard 0/1 label. Every pipeline here

    # supports predict_proba because SVC was built with probability=True.

    y_score = model.predict_proba(X_test)[:, 1]

 

    return {

        "Accuracy": accuracy_score(y_test, y_pred),

        "AUC": roc_auc_score(y_test, y_score),

        "Precision": precision_score(y_test, y_pred),

        "Recall": recall_score(y_test, y_pred),

        "F1": f1_score(y_test, y_pred),

        "MCC": matthews_corrcoef(y_test, y_pred),

    }

 

# =============================================================================

#  MAIN - tie everything together

# =============================================================================

 

def main():

    print("=" * 70)

    print(" Spambase - training 5 classification models")

    print("=" * 70)

 

    # --- Load data -----------------------------------------------------------

    X, y, feature_names = load_dataset()

    print("Dataset loaded: %d rows, %d features" % (X.shape[0], X.shape[1]))

    print("Class balance: %d not-spam (0), %d spam (1)"

          % (int((y == 0).sum()), int((y == 1).sum())))

 

    # --- Train / test split --------------------------------------------------

    # stratify=y keeps the same spam / not-spam ratio in both splits.

    X_train, X_test, y_train, y_test = train_test_split(

        X, y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y,

    )

    print("Train rows: %d | Test rows: %d" % (len(X_train), len(X_test)))

 

    # --- Make sure the output folder for models exists -----------------------

    os.makedirs(MODEL_DIR, exist_ok=True)

 

    # A mapping from display name to the file name we save the model under.

    model_files = {

        "Logistic Regression": "logistic_regression.joblib",

        "Decision Tree": "decision_tree.joblib",

        "KNN": "knn.joblib",

        "Naive Bayes": "naive_bayes.joblib",

        "Support Vector Machine": "svm.joblib",

    }

 

    # --- Train, evaluate and save every model --------------------------------

    models = build_models()

    all_metrics = {}   # collect metrics so we can print a comparison table

 

    for name, model in models.items():

        print("\nTraining: %s ..." % name)

        model.fit(X_train, y_train)                 # learn from the train set

        metrics = evaluate(model, X_test, y_test)   # score on the test set

        all_metrics[name] = metrics

 

        # Save the whole pipeline (scaler + classifier) to disk.

        out_path = os.path.join(MODEL_DIR, model_files[name])

        joblib.dump(model, out_path)

        print("  saved -> %s" % os.path.relpath(out_path, HERE))

 

    # --- Print a clean comparison table --------------------------------------

    print("\n" + "=" * 90)

    print(" MODEL COMPARISON (metrics computed on the held-out test set)")

    print("=" * 90)

    header = "%-24s %9s %9s %9s %9s %9s %9s" % (

        "Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")

    print(header)

    print("-" * len(header))

    for name, m in all_metrics.items():

        print("%-24s %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f" % (

            name, m["Accuracy"], m["AUC"], m["Precision"],

            m["Recall"], m["F1"], m["MCC"]))

 

    # --- Save feature names and reference metrics for the app ----------------

    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as fh:

        json.dump(feature_names, fh, indent=2)

 

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as fh:

        json.dump(all_metrics, fh, indent=2)

 

    # Also save the mapping of display-name -> file-name so the app can load it.

    with open(os.path.join(MODEL_DIR, "model_files.json"), "w") as fh:

        json.dump(model_files, fh, indent=2)

 

    # --- Save the test set to CSV (this is what you upload in the app) --------

    # We re-attach the target column so the uploaded CSV contains BOTH the input

    # features and the true label - the app needs the label to score the models.

    test_df = X_test.copy()

    test_df[TARGET_COLUMN] = y_test.values

    test_df.to_csv(TEST_CSV, index=False)

    print("\nTest data written to: %s  (%d rows)"

          % (os.path.relpath(TEST_CSV, HERE), len(test_df)))

 

    # --- Also print a ready-to-paste Markdown table for the README -----------

    print("\n" + "=" * 70)

    print(" Markdown comparison table (copy into README.md)")

    print("=" * 70)

    print("| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")

    print("| --- | --- | --- | --- | --- | --- | --- |")

    for name, m in all_metrics.items():

        print("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (

            name, m["Accuracy"], m["AUC"], m["Precision"],

            m["Recall"], m["F1"], m["MCC"]))

 

    print("\nDone. All models trained, evaluated and saved successfully.")

 

if __name__ == "__main__":

    main()
