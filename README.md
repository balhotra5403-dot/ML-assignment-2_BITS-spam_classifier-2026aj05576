README.md

# Spam E-mail Classification — ML Assignment 2

 

An end-to-end machine learning project that trains **five classification

models** on the UCI **Spambase** dataset, evaluates them with six metrics, and

serves the results through an interactive **Streamlit** web application.

 


> 

> - **GitHub Repository:** https://github.com/balhotra5403-dot/ML-assignment-2_BITS-spam_classifier-2026aj05576/tree/main/model

> - **Live Streamlit App:** https://apcnokzzu3ccv7h6hhmbi2.streamlit.app 

 

---

 

## a. Problem statement

 

E-mail spam (unsolicited commercial e-mail) wastes users' time and can carry

phishing or malware. The goal of this project is to **automatically classify an

e-mail as spam (1) or not-spam (0)** from numeric features describing the

e-mail's word and character frequencies. This is a **binary classification**

problem. We train five different classifiers, compare them on six evaluation

metrics, and expose the best models through a web app where a user can upload

test data and instantly see how each model performs.

 

## b. Dataset description

 

- **Source:** UCI Machine Learning Repository — *Spambase* dataset.

- **Instances (rows):** 4,601 e-mails.

- **Features (columns):** 57 numeric input features + 1 target label.

  - 48 `word_freq_*` features — percentage of words in the e-mail matching a

    given word (e.g. `free`, `money`, `business`).

  - 6 `char_freq_*` features — percentage of characters matching a given symbol

    (`;`, `(`, `[`, `!`, `$`, `#`).

  - 3 `capital_run_length_*` features — average / longest / total length of

    runs of capital letters.

- **Target:** `target` — `1` = spam, `0` = not-spam.

- **Class balance:** 2,788 not-spam (0) and 1,813 spam (1).

- **Train / test split:** 80% train (3,680 rows) / 20% test (921 rows),

  stratified to preserve the class ratio. The 921-row test set is saved as

  [`test_data.csv`](test_data.csv) and is the file to upload in the app.

 

The requirements (≥ 12 features and ≥ 500 instances) are comfortably met:

**57 features, 4,601 instances.**

 

## c. GitHub repository link

 

https://github.com/balhotra5403-dot/ML-assignment-2_BITS-spam_classifier-2026aj05576

 

## d. Models used

 

All five models are trained on the **same** dataset. Each model is a

scikit-learn `Pipeline` that first standard-scales the features and then applies

the classifier. Metrics below are computed on the held-out **test set**.

 

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9110 | 0.9078 | 0.8828 | 0.8926 | 0.8877 | 0.8140 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8328 | 0.9376 | 0.7146 | 0.9587 | 0.8188 | 0.6946 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

### Observations about model performance

| Model | Performance observation |
| --- | --- |
| Logistic Regression | **Best overall.** The spam features are high-dimensional and close to linearly separable, which suits a linear model. It scores highest on Accuracy, AUC, F1 and MCC with balanced precision and recall. |
| Decision Tree | Interpretable with reasonable accuracy, but has the **lowest AUC (0.9078)**. A single tree can produce coarse probability estimates and may overfit. |
| KNN | Performs well after scaling (Accuracy 0.9077, AUC 0.9506). It is sensitive to feature scaling and slower at prediction because it compares each test point with the training data. |
| Naive Bayes | **Lowest accuracy and precision**, likely because word-frequency features are correlated. It has the **highest recall (0.9587)**, catching most spam at the cost of more false positives. |
| Support Vector Machine | A close second to Logistic Regression and has the **highest precision (0.9277)**. It has strong AUC and F1, with higher training and prediction cost. |
| **Overall winner** | **Logistic Regression** leads in Accuracy, AUC, F1 and MCC, with balanced precision and recall. |

 

---

 

## Repository structure

 

```

ML_assignment_2/

├── app.py                 # Streamlit web application (loads models, shows metrics)

├── train_models.py        # Trains & saves the 5 models, writes test_data.csv

├── requirements.txt       # Pinned library versions (match the saved models)

├── README.md              # This file

├── test_data.csv          # 921-row test set to upload in the app

├── spambase.data          # Raw UCI dataset (no header)

├── spambase.names         # Official column descriptions

└── model/                 # Saved trained pipelines + metadata

    ├── logistic_regression.joblib

    ├── decision_tree.joblib

    ├── knn.joblib

    ├── naive_bayes.joblib

    ├── svm.joblib

    ├── feature_names.json  # ordered list of the 57 feature names

    ├── model_files.json    # display name -> model file name

    └── metrics.json        # reference metrics measured during training

```

 

## How to run locally

# 1. Create and activate a virtual environment (Python 3.11 recommended)

python -m venv .venv

source .venv/bin/activate        # Windows: .venv\\Scripts\\activate

# 2. Install the exact dependencies

pip install -r requirements.txt

# 3. (Optional) re-train the models — this regenerates model/ and test_data.csv

python train_models.py

# 4. Launch the web app

streamlit run app.py

Then open the URL shown in the terminal, upload `test_data.csv`, choose a model,

and view the metrics, confusion matrix and classification report.

 

## Streamlit app features

 

- **CSV upload** of test data (sidebar).

- **Model selection dropdown** (each model, plus a *Compare all models* view).

- **Display of all six evaluation metrics** for the chosen model.

- **Confusion matrix** (heatmap) and **classification report** table.

 

The app only **loads** the pre-trained models and evaluates them on the uploaded

CSV — it never re-trains on the uploaded data.





