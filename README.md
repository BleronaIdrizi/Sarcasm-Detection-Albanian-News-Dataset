# Sarcasm-Detection-Albanian-News-Dataset

---

## Environment Setup

It is recommended to run this project inside a Python virtual environment to avoid dependency conflicts.

Create and activate a virtual environment from the project root:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:
```
pip install -r requirements.txt
```

Set the project interpreter to the virtual environment: .venv/bin/python

---

## Dataset Download

This project uses the Kosovo News Articles Dataset from Kaggle.
The dataset is downloaded programmatically to ensure reproducibility and to avoid storing raw data in the repository.
The dataset download script:
- Fetches the dataset from Kaggle using kagglehub
- Automatically locates the CSV file
- Saves it locally to the data/ directory
- Skips downloading if the dataset already exists

### How to Run

The dataset download is implemented as a Jupyter Notebook.

1. Open the notebook:
   scripts/download_dataset.ipynb

2. Make sure the correct Python virtual environment is selected as the kernel.

3. Execute all cells in the notebook to download the dataset.

---

### Output

After successful execution, the dataset will be saved to: data/kosovo_news.csv. This step must be completed before running preprocessing or modeling notebooks.

---