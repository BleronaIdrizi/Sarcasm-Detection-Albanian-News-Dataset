# Sarcasm Detection in Albanian News

## Overview

This project explores sarcasm detection in Albanian news, a low-resource NLP setting where labeled data is limited and context matters a lot. It covers the full workflow from raw news preprocessing and dataset construction to model training, evaluation, and comparison. All sarcasm models are now evaluated with **5-fold stratified cross-validation**, reporting each metric as mean ± standard deviation across folds. The best current sarcasm model is **TF-IDF + LinearSVC**, reaching **83.05% accuracy** and **83.03% macro F1-score** on the balanced sarcasm dataset.

## Highlights

- End-to-end Albanian NLP pipeline, from cleaning raw articles to evaluating sarcasm detection models.
- Large-scale preprocessing of the Kosovo news corpus, starting from more than **2.5M raw rows**.
- Custom balanced sarcasm dataset with **2,732 examples**, built from Kosovo news and FLOSSK historical sources.
- Classical ML baselines with **TF-IDF**, **LinearSVC**, **Logistic Regression**, and **Multinomial Naive Bayes**.
- Multilingual transformer experiments with **XLM-RoBERTa** and **DistilBERT multilingual**.
- Clear comparison of models using accuracy, F1, precision, and recall.
- Reproducible notebook-based workflow with reusable helper scripts.

## Tech Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, TF-IDF, LinearSVC, Logistic Regression, MultinomialNB
- **Deep learning / NLP:** PyTorch, HuggingFace Transformers, HuggingFace Datasets
- **Transformer models:** XLM-RoBERTa, DistilBERT multilingual
- **Visualization:** matplotlib, seaborn
- **Experiment workflow:** Jupyter Notebook / JupyterLab
- **Dataset tooling:** KaggleHub, FLOSSK historical source utilities
- **Annotation support:** OpenAI-assisted labeling workflow

## Results Summary

### Best Sarcasm Detection Model

| Task | Best Model | Accuracy | F1 Score |
|------|------------|---------:|---------:|
| Sarcasm Detection | TF-IDF + LinearSVC | **0.8305** | **0.8303 macro F1** |
| Category Detection | TF-IDF + LinearSVC | **0.7237** | **0.6994 weighted F1** |

The strongest sarcasm detection result comes from a classical linear model rather than a transformer. In this dataset version, **TF-IDF + LinearSVC** outperforms Logistic Regression, MultinomialNB, and multilingual DistilBERT.

## Example Usage

> **Note:** The repository currently ships the fine-tuned DistilBERT checkpoints under `models/`. The TF-IDF vectorizer and LinearSVC artifacts shown below are produced by training and exporting them from `notebooks/03_sarcasm_with_annotation/03d_sarcasm_models_and_evaluation.ipynb` (see Future Work).

After exporting, inference can be run as follows:

```python
from joblib import load

vectorizer = load("models/tfidf_vectorizer.joblib")
model = load("models/sarcasm_linearsvc.joblib")

text = "Premtimet u realizuan aq shpejt sa qytetaret ende po i presin."

features = vectorizer.transform([text])
prediction = model.predict(features)[0]

label = "sarcastic" if prediction == 1 else "non-sarcastic"
print(label)
```

Expected label space:

- `1`: sarcastic text
- `0`: non-sarcastic text

## Environment Setup

It is recommended to use a Python virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Jupyter notebooks, make sure the selected kernel/interpreter points to the virtual environment.

## Dataset

The raw corpus is the [Kosovo News Articles Dataset](https://www.kaggle.com/datasets/gentrexha/kosovo-news-articles-dataset) (Kaggle, v4), complemented by historical Albanian-language sources from the [FLOSSK](https://flossk.org/) digital archive. The raw dataset is downloaded from Kaggle using:

```text
scripts/download_dataset.ipynb
```

After download, the raw file is expected at:

```text
data/kosovo_news.csv
```

## Data Preprocessing

This step, implemented in `notebooks/00_preprocess.ipynb`, prepares the raw Kosovo news dataset for modeling.

Initial dataset:

- Rows: **2,593,450**
- Columns: `content`, `date`, `title`, `category`, `author`, `source`

Main preprocessing steps:

- Removed rows with missing values in relevant columns
- Removed duplicate records
- Dropped unused metadata columns (`date`, `author`)
- Created a unified `text` field from article title and content
- Applied basic text normalization, including lowercasing, URL removal, and whitespace cleanup
- Inspected category labels and category distribution

Important data quality findings:

| Item | Count |
|------|------:|
| Missing `content` values | 10,409 |
| Missing `category` values | 26,551 |
| Duplicate rows | 613,866 |
| Unique category combinations before filtering | 1,235 |

Final preprocessed dataset:

- Rows: **1,455,862**
- Columns: `content`, `title`, `category`, `source`, `text`
- Missing values after preprocessing: **0**
- Duplicate rows after preprocessing: **0**

Saved output:

```text
data/preprocessed_kosovo_news.csv
```

## Category Detection

This step, implemented in `notebooks/01_category_detection.ipynb`, evaluates category classification as an intermediate NLP task before sarcasm detection.

Two label settings were prepared:

- **Multi-Class Dataset**: original category combinations filtered to classes with at least 50 samples
- **Primary Category Dataset**: category labels reduced to one primary category, then filtered to classes with at least 50 samples

Class filtering summary:

| Dataset | Classes Before | Classes After |
|---------|---------------:|--------------:|
| Multi-Class Dataset | 1,235 | 274 |
| Primary Category Dataset | 252 | 181 |

Prepared dataset sizes:

| Dataset | Rows |
|---------|-----:|
| Multi-Class Dataset | 1,449,457 |
| Primary Category Dataset | 1,454,658 |

## Modeling

Classical models were evaluated with **TF-IDF features** and **5-fold Stratified K-Fold cross-validation**:

- `LinearSVC`
- `SGDClassifier` configured as logistic regression
- `MultinomialNB`

A transformer experiment was also run with `xlm-roberta-base`. Because full 5-fold transformer training is computationally expensive, it was evaluated using a stratified train/validation split on a subset of 50,000 training samples and 10,000 validation samples.

## Results

Weighted F1-score and accuracy were used as the main comparison metrics.

### Primary Category Dataset - 181 classes

| Model | Accuracy | Weighted F1 |
|-------|---------:|------------:|
| LinearSVC | **0.7237** | **0.6994** |
| SGD_LogReg | 0.6558 | 0.6044 |
| XLM-RoBERTa | 0.6061 | 0.5570 |
| MultinomialNB | 0.5407 | 0.5229 |

### Multi-Class Dataset - 274 classes

| Model | Accuracy | Weighted F1 |
|-------|---------:|------------:|
| LinearSVC | **0.6975** | **0.6682** |
| SGD_LogReg | 0.6258 | 0.5681 |
| XLM-RoBERTa | 0.5619 | 0.4970 |
| MultinomialNB | 0.5210 | 0.4912 |

## Key Findings

- `LinearSVC` is the strongest baseline so far in both category settings.
- Performance is higher on the primary-category setup than on the original multi-class setup, which is expected because the 274-class setting is more complex and noisier.
- TF-IDF with linear models works well for Albanian news category classification and provides a strong reference baseline.
- The transformer experiment is useful as a contextual deep learning baseline, but in this run it was limited by subset size and CPU training.
- Error analysis for `LinearSVC` shows strong recall for clearer categories such as Sport, Ndërkombëtare, Maqedoni, and Yjet, while broader or semantically overlapping labels such as Lajme, Bota, Kosovë, Shqipëri, and Shkurt are more often confused.

## Sarcasm Detection Dataset Construction

The sarcasm detection workflow is implemented under:

```text
notebooks/03_sarcasm_with_annotation/
```

This part of the project prepares a binary sarcasm dataset for Albanian news text. The label space is:

- `1`: sarcastic text
- `0`: non-sarcastic text

The workflow is split into four notebooks to keep annotation, historical-source synchronization, dataset construction, and model evaluation reproducible and easier to audit.

### 03a - Sarcasm Dataset Preparation and Annotation

Implemented in:

```text
notebooks/03_sarcasm_with_annotation/03a_sarcasm_prepare_dataset.ipynb
```

This notebook prepares the Kosovo news dataset for sarcasm annotation. It uses the preprocessed news file as input and stores annotation progress in a resume-friendly format so long labeling runs can be stopped and continued safely.

Main responsibilities:

- Load `data/preprocessed_kosovo_news.csv`
- Keep the relevant modeling columns (`content`, `category`, `source`, `is_sarcasm`, and confidence metadata when available)
- Apply OpenAI-assisted sarcasm labeling only to rows with missing labels
- Save annotation progress to `data/preprocessed_kosovo_news_with_is_sarcasm_v1.csv`

Current labeled Kosovo news file:

| File | Rows | Sarcastic | Non-sarcastic |
|------|-----:|----------:|--------------:|
| `preprocessed_kosovo_news_with_is_sarcasm_v1.csv` | 120,000 | 766 | 119,234 |

### 03b - FLOSSK Historical Sources Sync

Implemented in:

```text
notebooks/03_sarcasm_with_annotation/03b_flossk_historical_sources_sync.ipynb
```

This notebook synchronizes historical Albanian-language sources from the FLOSSK digital books/newspapers platform. The utility logic is stored separately in:

```text
scripts/03b_flossk_historical_sources_utils.py
```

The FLOSSK workflow has two parts:

- `gazetat/`: searchable historical newspaper archive used to build a heuristic sarcasm bootstrap dataset
- `librat/`: books catalog metadata export

Important methodological note: FLOSSK labels are heuristic bootstrap labels, not final manually verified gold annotations. They are used to enrich the sarcasm dataset with additional positive and negative examples.

Main outputs:

```text
data/sarcasm_flossk_historic_bootstrap.csv
data/flossk_books_catalog.csv
```

Current FLOSSK bootstrap file:

| File | Rows | Sarcastic | Non-sarcastic |
|------|-----:|----------:|--------------:|
| `sarcasm_flossk_historic_bootstrap.csv` | 1,200 | 600 | 600 |

### 03c - Merge and Rebalance Dataset

Implemented in:

```text
notebooks/03_sarcasm_with_annotation/03c_merge_and_rebalance_dataset.ipynb
```

This notebook builds the final balanced sarcasm dataset used for model evaluation. It merges the OpenAI-labeled Kosovo news file with the FLOSSK bootstrap dataset, normalizes label formats, removes duplicate text, and balances the final dataset to a 50/50 class distribution.

Inputs:

```text
data/preprocessed_kosovo_news_with_is_sarcasm_v1.csv
data/sarcasm_flossk_historic_bootstrap.csv
```

Output:

```text
data/preprocessed_kosovo_news_with_is_sarcasm.csv
```

Final balanced dataset:

| File | Rows | Sarcastic | Non-sarcastic |
|------|-----:|----------:|--------------:|
| `preprocessed_kosovo_news_with_is_sarcasm.csv` | 2,732 | 1,366 | 1,366 |

### 03d - Sarcasm Models and Evaluation

Implemented in:

```text
notebooks/03_sarcasm_with_annotation/03d_sarcasm_models_and_evaluation.ipynb
```

This notebook trains and evaluates sarcasm detection models on the balanced dataset produced by `03c_merge_and_rebalance_dataset.ipynb`.

Baseline models:

- `MultinomialNB`
- `LinearSVC`
- `LogisticRegression`

Transformer model:

- `distilbert-base-multilingual-cased`

The results table uses the shared leaderboard helper:

```text
scripts/leaderboard.py
```

Metrics reported:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score

All models are evaluated with **5-fold stratified cross-validation**. The TF-IDF vectorizer is refitted inside each fold on the training portion only (to avoid vocabulary leakage), and the transformer is fine-tuned from scratch on each fold. Each metric is reported as the **mean ± standard deviation across the 5 folds**.

Current results on the balanced sarcasm dataset (mean ± standard deviation over 5 folds):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|---------:|----------------:|-------------:|---------:|
| LinearSVC | **0.8305 ± 0.0048** | **0.8325 ± 0.0044** | **0.8305 ± 0.0047** | **0.8303 ± 0.0048** |
| MultinomialNB | 0.8133 ± 0.0070 | 0.8137 ± 0.0073 | 0.8133 ± 0.0070 | 0.8133 ± 0.0070 |
| LogisticRegression | 0.8130 ± 0.0058 | 0.8171 ± 0.0054 | 0.8130 ± 0.0057 | 0.8123 ± 0.0059 |
| DistilBERT multilingual | 0.7972 ± 0.0074 | 0.7986 ± 0.0081 | 0.7972 ± 0.0074 | 0.7970 ± 0.0073 |

The best-performing sarcasm detection model is `LinearSVC`, achieving **83.05% accuracy** and **83.03% macro F1-score** averaged over the 5 folds. The fold-to-fold variation is small, so this result is stable rather than an artifact of one particular split. This confirms that TF-IDF features with a linear classifier provide a strong baseline for the balanced sarcasm dataset.

The transformer model (`distilbert-base-multilingual-cased`) performs below the classical linear baselines in this experiment. This may be influenced by the relatively small balanced dataset size, the heuristic nature of part of the FLOSSK bootstrap data, and the computationally constrained fine-tuning setup. The transformer result remains useful as a contextual baseline, but the current evidence suggests that the TF-IDF + LinearSVC approach is the strongest model for this dataset version.

Confusion-matrix analysis of the best model uses the **aggregated out-of-fold predictions**, so every example in the dataset is counted exactly once using the prediction from the fold where it was held out. `LinearSVC` is slightly stronger at recognizing **non-sarcastic** texts than **sarcastic** ones. It correctly classifies **1,185 of 1,366 non-sarcastic examples (86.7%)**, while **181 cases (13.3%)** are incorrectly predicted as sarcastic.

For the **sarcastic** class, the model correctly identifies **1,084 of 1,366 examples (79.4%)**, while **282 cases (20.6%)** are misclassified as non-sarcastic. This suggests that the remaining errors are concentrated more heavily on sarcastic texts, which is consistent with the higher contextual ambiguity of sarcasm in Albanian news language.

Because every model now uses 5-fold cross-validation, the leaderboard reports non-zero standard deviation columns, matching the evaluation protocol used in the category detection experiments. Out-of-fold predictions from the transformer run are also saved to `data/sarcasm_balanced_predictions_v1.csv`.

## Limitations

- The final balanced sarcasm dataset is relatively small, with **2,732 rows**, which limits deep learning fine-tuning.
- Part of the sarcasm dataset comes from FLOSSK heuristic bootstrap labels, so labeling quality may vary.
- Transformer experiments were computationally constrained and not fully optimized with extensive GPU training.
- Sarcasm is highly context-dependent, especially in news and political language, so some labels may be ambiguous.

## Future Work

- Fine-tune transformer models on GPU with larger batches, more epochs, and better hyperparameter search.
- Expand the manually reviewed sarcasm dataset to improve label quality and reduce heuristic noise.
- Add multiple random seeds on top of the current 5-fold cross-validation for even more robust sarcasm evaluation.
- Export the best trained model and vectorizer into `models/` for direct inference.
- Add a lightweight API or Streamlit demo for interactive sarcasm prediction.
- Include richer error analysis, such as confusion matrices and examples of false positives/false negatives.

## Project Structure

```text
Sarcasm-Detection-Albanian-News-Dataset/
├── data/
│   ├── kosovo_news.csv
│   ├── preprocessed_kosovo_news.csv
│   └── ...
├── models/
├── notebooks/
│   ├── 00_preprocess.ipynb
│   ├── 01_category_detection.ipynb
│   └── 03_sarcasm_with_annotation/
│       ├── 03a_sarcasm_prepare_dataset.ipynb
│       ├── 03b_flossk_historical_sources_sync.ipynb
│       ├── 03c_merge_and_rebalance_dataset.ipynb
│       └── 03d_sarcasm_models_and_evaluation.ipynb
├── scripts/
│   ├── 03b_flossk_historical_sources_utils.py
│   ├── download_dataset.ipynb
│   └── leaderboard.py
├── requirements.txt
└── README.md
```

## Current Status

The project currently includes:

- A clean preprocessing pipeline for the Kosovo news dataset
- A category classification benchmark with classical and transformer-based models
- A labeled sarcasm annotation workflow
- FLOSSK historical source synchronization for heuristic sarcasm bootstrap examples
- A final 50/50 balanced sarcasm dataset with **2,732 rows**
- A sarcasm modeling notebook with TF-IDF baselines and a multilingual DistilBERT experiment, all evaluated with 5-fold stratified cross-validation

For category detection, the best result so far is achieved by `LinearSVC` on the primary-category dataset with **72.37% accuracy** and **69.94% weighted F1-score**.

For sarcasm detection, the best current result is achieved by `LinearSVC` on the final balanced dataset with **83.05% accuracy** and **83.03% macro F1-score** (mean over 5 folds).

## Citation

This repository accompanies the master's thesis:

> Blerona Idrizi, *"Detektimi i sarkazmës dhe klasifikimi i artikujve të lajmeve sipas kategorive në 'Albanian News Dataset' duke përdorur Transformer Models"* (Sarcasm Detection and News Article Category Classification in the Albanian News Dataset using Transformer Models), Master's thesis, Faculty of Electrical and Computer Engineering, University of Prishtina, 2026.

```bibtex
@mastersthesis{idrizi2026sarcasm,
  author = {Idrizi, Blerona},
  title  = {Detektimi i sarkazm{\"e}s dhe klasifikimi i artikujve t{\"e} lajmeve sipas kategorive n{\"e} "Albanian News Dataset" duke p{\"e}rdorur Transformer Models},
  school = {University of Prishtina, Faculty of Electrical and Computer Engineering},
  year   = {2026}
}
```

## Acknowledgements

- [Kosovo News Articles Dataset](https://www.kaggle.com/datasets/gentrexha/kosovo-news-articles-dataset) by Gent Rexha (Kaggle).
- [FLOSSK](https://flossk.org/) for the historical Albanian newspaper and book archive used in the bootstrap dataset.
- Part of the sarcasm annotation used an OpenAI-assisted labeling workflow; heuristic FLOSSK bootstrap labels are documented as such throughout.
