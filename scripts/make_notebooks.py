"""Build and execute the EDA, training and inference notebooks. Run from repo root: python scripts/make_notebooks.py"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, NB / name)
    print("wrote", name)


SETUP = "import sys\nsys.path.insert(0, '.')\n%matplotlib inline\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n" \
        "from src import config\nfrom src.data import load_clean\ndf = load_clean()"

build("01_eda.ipynb", [
    ("md", "# 01 - Exploratory Data Analysis\nHeart Disease UCI (Cleveland). Run `python -m src.data` first."),
    ("code", SETUP + "\nprint(df.shape)\ndf.head()"),
    ("code", "df.describe().T"),
    ("md", "## Class balance"),
    ("code", "sns.countplot(x=config.TARGET, data=df)\nplt.title('Class balance')\n"
             "print(df[config.TARGET].value_counts(normalize=True))"),
    ("md", "## Numeric distributions"),
    ("code", "df[config.NUMERIC].hist(bins=20, figsize=(12, 7))\nplt.show()"),
    ("md", "## Correlation heatmap"),
    ("code", "plt.figure(figsize=(10, 8))\nsns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')\nplt.show()"),
    ("md", "## Categorical features versus target"),
    ("code", "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
             "for ax, col in zip(axes, ['cp', 'thal', 'ca']):\n"
             "    sns.countplot(x=col, hue=config.TARGET, data=df, ax=ax)\nplt.show()"),
])

build("02_training.ipynb", [
    ("md", "# 02 - Feature engineering, model development and MLflow tracking\n"
           "Preprocessing (StandardScaler + OneHotEncoder) and the estimator form a single Pipeline. "
           "Two models are tuned with GridSearchCV and compared by 5-fold CV ROC-AUC."),
    ("code", "import sys, json\nsys.path.insert(0, '.')\nimport pandas as pd\nfrom src.train import train_all, CANDIDATES\n"
             "from src import config\nCANDIDATES"),
    ("code", "best, results = train_all()\nprint('Best model:', best)"),
    ("code", "pd.DataFrame(results).T.drop(columns='params').round(3)"),
    ("md", "Logistic Regression is selected on CV ROC-AUC. Runs, plots and models are tracked in the MLflow "
           "experiment `heart-disease` (`mlflow ui`)."),
])

build("03_inference.ipynb", [
    ("md", "# 03 - Inference with the saved pipeline"),
    ("code", "import sys\nsys.path.insert(0, '.')\nimport joblib, pandas as pd\nfrom src import config\n"
             "model = joblib.load(config.MODEL_PATH)\n"
             "patient = {'age': 63, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 233, 'fbs': 1, 'restecg': 2,\n"
             "           'thalach': 150, 'exang': 0, 'oldpeak': 2.3, 'slope': 3, 'ca': 0, 'thal': 6}\n"
             "proba = model.predict_proba(pd.DataFrame([patient])[config.FEATURES])[0, 1]\n"
             "print(f'Disease probability: {proba:.3f} -> ' + ('disease' if proba >= 0.5 else 'no disease'))"),
])
