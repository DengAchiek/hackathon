# Mteja learning notebooks in Python

These are readable **`.py` Python files**, using `# %%` for code cells and
`# %% [markdown]` for explanations. They have no `.ipynb` JSON wrapper.
The original Jupyter notebooks are preserved in the project.

| Python notebook | Purpose | Default practice data |
| --- | --- | --- |
| `GhostDemand_Training_Improved.py` | EDA, grouped splits, TF-IDF + logistic regression, evaluation, matching, extraction, candidate model export and local API checks | Embedded synthetic customer enquiries and fictional catalog |
| `GhostDemand_Forecasting.py` | Demand forecasts, backtesting and stock recommendations | Synthetic requests, sales and stock generated in Python |
| `PesaShield_AI_Roadmap.py` | SMS data checks, EDA, grouped split, TF-IDF + logistic regression, evaluation and local API checks | Included `data/pesashield_synthetic_mpesa_sms.csv` |

## 1. Install in a separate learning environment

Use Python 3.11 or newer; validation uses Python 3.12. Extract the whole ZIP
and open a terminal inside `Mteja_Python_Notebooks`, or use this repository's
`notebooks/python` directory. Do not use the live app's virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

## 2. Run as Python notebook cells

1. Open this folder in VS Code.
2. Use the Python and Jupyter extensions.
3. Open one of the `.py` files and select the `.venv` Python interpreter.
4. Click **Run Cell** above `# %%`, or **Run All** to execute the complete workflow.
5. Tables and plots appear in the Python Interactive window. Read explanations
   in the markdown cells alongside the code.

Run the setup cell first. **Restart the kernel before switching notebooks**:
their variables, data and model definitions are intentionally independent.
All three default workflows run on CPU. Transformer experiments in the demand
notebook remain optional and disabled; they require their own extra packages
and model downloads if you deliberately enable them.

## 3. Or run as regular Python scripts

```bash
python GhostDemand_Training_Improved.py
python GhostDemand_Forecasting.py
python PesaShield_AI_Roadmap.py
```

Tables print in the terminal, and charts are saved as PNGs. Each script writes
only to its own folder under `runs/`, never to the application's active models,
configuration or customer database. Local API examples use test clients; they
do not start a public service or connect accounts.

## 4. Supply your own reviewed data

- **Demand classification:** change `LABELED_REQUESTS_CSV` in the setup cell to
  the absolute path of your reviewed dataset. Keep the notebook's required
  columns and six reason labels. Optional public-source inspection uses `data/`
  only after you enable `INSPECT_PUBLIC_DATA`.
- **Forecasting:** put `requests.csv`, and optionally `sales.csv` and `stock.csv`,
  in `data/`. Their required columns are explained in the Python notebook.
- **PesaShield:** the included CSV is synthetic. Review `data/PesaShield_Dataset_README.md`
  before replacing it with lawful, consented and appropriately anonymised data.

Practice scores do not establish performance on real customers. PesaShield
assesses suspicious text patterns; it does not verify that money arrived.

## 5. Open in JupyterLab or Google Colab

An `.ipynb` file is always JSON internally, even when its cells contain Python.
Use the provided `.py` files for plain code. If you specifically need a Jupyter
or Colab file, convert a copy with Jupytext:

```bash
python -m pip install jupytext
jupytext --to ipynb GhostDemand_Training_Improved.py
```

Open the resulting notebook normally, rather than in a text editor. Its first
cell assumes the current directory contains this package's `data/` folder when
`__file__` is unavailable. On Colab, upload/extract the whole package and change
to that folder before running cells. Keep the original `.py` file for readable
source code.

Model exports and API responses may still use JSON as a data interchange format;
that is separate from how the training code itself is stored.
