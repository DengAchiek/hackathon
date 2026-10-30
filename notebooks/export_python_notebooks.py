"""Export the learning notebooks as runnable Python files with # %% cells.

Run with nbformat and jupytext installed. This only prepares learning files;
it does not import the app, replace its models, or change its configuration.
"""
import ast
from pathlib import Path
import shutil
from textwrap import dedent
import zipfile

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "python"
SOURCES = (
    ROOT / "notebooks" / "GhostDemand_Training_Improved.ipynb",
    ROOT / "notebooks" / "GhostDemand_Forecasting.ipynb",
    ROOT / "PesaShield_AI" / "PesaShield_AI_Roadmap.ipynb",
)

SETUP = '''\
from pathlib import Path
import os
from IPython import get_ipython
from IPython.display import display

# In VS Code, open this file and use Run Cell or Run All. In Jupyter, first
# change to the folder containing the Python notebooks. Python scripts also work.
NOTEBOOK_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
                else globals().get("NOTEBOOK_DIR", Path.cwd().resolve()))
DATA_DIR = NOTEBOOK_DIR / "data"
RUN_DIR = NOTEBOOK_DIR / "runs" / "__RUN_NAME__"
RUN_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(RUN_DIR)
os.environ.setdefault("MPLCONFIGDIR", str(RUN_DIR / ".matplotlib"))

IN_NOTEBOOK = get_ipython() is not None
import matplotlib
if IN_NOTEBOOK:
    get_ipython().run_line_magic("matplotlib", "inline")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

_plot_number = 0
def show_plot():
    """Display charts interactively, or save them when run with python."""
    global _plot_number
    if IN_NOTEBOOK:
        plt.show()
        return
    for number in plt.get_fignums():
        _plot_number += 1
        path = RUN_DIR / f"figure_{_plot_number:02d}.png"
        plt.figure(number).savefig(path, bbox_inches="tight", dpi=140)
        print("Saved chart:", path)
    plt.close("all")

print("Learning outputs:", RUN_DIR)
'''


def python_source(source, notebook_name):
    """Convert only notebook-specific execution syntax, keeping model logic."""
    if source.startswith("%%writefile forecasting.py\n"):
        source = source.partition("\n")[2]
        source = source.replace("from __future__ import annotations\n", "")
        # The exported notebook defines helpers directly, instead of writing
        # over forecasting.py or running that module's separate CLI demo.
        source = source.partition('\nif __name__ == "__main__":')[0].rstrip()

    if notebook_name == "GhostDemand_Training_Improved":
        source = source.replace("INSTALL_MISSING_CORE = True", "INSTALL_MISSING_CORE = False")
        source = source.replace("INSPECT_PUBLIC_DATA = True", "INSPECT_PUBLIC_DATA = False")
        source = source.replace(
            "PUBLIC_DATA_DIR = PROJECT_ROOT.parent if PROJECT_ROOT else Path.cwd()",
            "PUBLIC_DATA_DIR = DATA_DIR",
        )
        # The export is standalone. App compatibility remains a separate step
        # in the original notebook; it must not import live application code.
        start = "PROJECT_ROOT = next((p for p in [Path.cwd(), *list(Path.cwd().parents)[:3]]\n"
        end = "                     if (p/'training_data.json').exists() and (p/'demand_ml.py').exists()), None)"
        source = source.replace(start + end, "PROJECT_ROOT = None  # Standalone learning run; no application imports.")
        source = source.replace(
            "get_ipython().run_line_magic('matplotlib', 'inline')",
            "# The editor manages inline plotting; normal Python saves charts below.",
        )

    if notebook_name == "GhostDemand_Forecasting":
        source = source.replace(
            "from forecasting import Config, forecast_products, make_demo_data, prepare_daily",
            "# Forecasting helpers were defined in the preceding Python cell.",
        )
        for name in ("requests.csv", "sales.csv", "stock.csv"):
            source = source.replace(f'os.path.exists("{name}")', f'(DATA_DIR / "{name}").exists()')
            source = source.replace(f'pd.read_csv("{name}")', f'pd.read_csv(DATA_DIR / "{name}")')

    if notebook_name == "PesaShield_AI_Roadmap":
        source = source.replace(
            'CSV_NAME = "pesashield_synthetic_mpesa_sms.csv"',
            'CSV_NAME = str(DATA_DIR / "pesashield_synthetic_mpesa_sms.csv")',
        )
        source = source.replace(
            "print('Run: %pip install fastapi httpx uvicorn and rerun this cell')",
            "print('Install requirements.txt, then rerun this cell.')",
        )
        # The source notebook stores this API example as one escaped string.
        # Show readable, editable Python across multiple lines in the export.
        for node in ast.parse(source).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "api_source"
                    and isinstance(node.value, ast.Constant)):
                api_source = ast.literal_eval(node.value)
                assert "'''" not in api_source
                old = ast.get_source_segment(source, node)
                source = source.replace(old, "api_source = r'''\n" + api_source.rstrip() + "\n'''")

    # Bare final DataFrame expressions display in notebooks, but normal Python
    # ignores them. Keep previews available in both execution modes.
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        value = tree.body[-1].value
        if (isinstance(value, ast.Name) or (isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr in {"head", "drop"})):
            node = tree.body[-1]
            lines = source.splitlines(keepends=True)
            expression = ast.get_source_segment(source, node)
            lines[node.lineno - 1:node.end_lineno] = [f"display({expression})\n"]
            source = "".join(lines).rstrip()

    has_plots = any(item in source for item in ("plt.subplots(", "plt.figure(", ".plot(cmap="))
    source = source.replace("plt.show()", "show_plot()")
    if has_plots and "show_plot()" not in source:
        source += "\nshow_plot()"
    ast.parse(source)
    return source


def export_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    intro = nbformat.v4.new_markdown_cell(dedent(f"""\
        # Python cell notebook: {path.stem.replace('_', ' ')}

        This is ordinary Python, arranged into `# %%` notebook cells.
        Open it in VS Code with the Python and Jupyter extensions and select
        **Run Cell** or **Run All**, or run `python {path.stem}.py`.
        Install `requirements.txt` first. See `README.md` for the full setup.

        Source: `{path.relative_to(ROOT)}`. Original explanations and model
        logic are retained. Outputs go to a separate `runs/{path.stem}/`
        folder. The scripts use practice data and never deploy a live model.
        """).strip())
    cells = [intro, nbformat.v4.new_code_cell(SETUP.replace("__RUN_NAME__", path.stem))]
    for cell in notebook.cells:
        if cell.cell_type == "raw":
            cell = nbformat.v4.new_markdown_cell(cell.source)
        if cell.cell_type == "code":
            cell.source = python_source(cell.source, path.stem)
            cell.outputs = []
            cell.execution_count = None
        elif path.stem == "GhostDemand_Training_Improved":
            cell.source = cell.source.replace(
                "Expand the collapsed supporting cells to inspect the embedded data or generated\nAPI source; the main training/evaluation code remains visible.",
                "All supporting data and API code are visible as ordinary Python cells.",
            ).replace(
                "On Colab, upload input files to the runtime and set `PUBLIC_DATA_DIR` to that\nfolder. On a local checkout, this cell recognizes the GhostDemand folder and\nits parent Downloads folder. Output stays in a separate learning directory.",
                "Use absolute paths for your own input files. Optional public-source inspection\nreads from this package's `data/` folder. Output stays in the separate run directory.",
            )
        elif path.stem == "GhostDemand_Forecasting":
            cell.source = cell.source.replace(
                "The next cell saves the forecasting code as `forecasting.py` next to the notebook, so it can be imported and reused in your app.",
                "The next cell defines the forecasting helpers directly in Python. It does not write or replace an application module.",
            )
        cell.metadata = {}
        cells.append(cell)
    result = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    target = DESTINATION / (path.stem + ".py")
    jupytext.write(result, target, fmt="py:percent")
    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    restored = jupytext.read(target)
    nbformat.validate(restored)
    assert [(c.cell_type, c.source.strip()) for c in restored.cells] == [
        (c.cell_type, c.source.strip()) for c in result.cells
    ], f"Cell round-trip differs: {path.name}"
    print("Exported:", target.relative_to(ROOT))


def package_exports():
    target = ROOT / "output" / "notebooks" / "Mteja_Python_Notebooks.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    files = [Path(path.stem + ".py") for path in SOURCES]
    files += [Path("README.md"), Path("requirements.txt"), Path(".gitignore")]
    files += [Path("data") / name for name in (
        "pesashield_synthetic_mpesa_sms.csv", "PesaShield_Dataset_README.md")]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(DESTINATION / relative, Path("Mteja_Python_Notebooks") / relative)
    print("Download bundle:", target.relative_to(ROOT))


if __name__ == "__main__":
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "data").mkdir(exist_ok=True)
    for source in SOURCES:
        export_notebook(source)
    for name in ("pesashield_synthetic_mpesa_sms.csv", "PesaShield_Dataset_README.md"):
        shutil.copy2(ROOT / "PesaShield_AI" / name, DESTINATION / "data" / name)
    package_exports()
