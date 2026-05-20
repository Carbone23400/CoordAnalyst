.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/Cheminfo.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/Cheminfo
    .. image:: https://readthedocs.org/projects/Cheminfo/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://Cheminfo.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/Cheminfo/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/Cheminfo
    .. image:: https://img.shields.io/pypi/v/Cheminfo.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/Cheminfo/
    .. image:: https://img.shields.io/conda/vn/conda-forge/Cheminfo.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/Cheminfo
    .. image:: https://pepy.tech/badge/Cheminfo/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/Cheminfo
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/Cheminfo

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

========
CoordAnalyst
========

## Description
    Add a short description here!
An educational tool helping students predict oxidation, geometry, number of d electrons, and other properties of a coordination complex as well as visualize it in 2D, 3D and draw a reliable IR and Raman spectra from streches in databases for common ligands

A longer description of your project goes here...
Our package named CoordAnalyst is based on the analysis of coordinate complexes. Indeed, the goal was to implement an educational tool to better understand complexes, by having the basics information. Thus, a user can enter the name or the formula of the complex on the interface to obtain its name and formula, the metal and its oxydation state, the geometry, the number of d electron and the coordination state. The list of the majors common ligands that are supported is displayed, to prevent any unknown input from the user. Furthermore, the 2D and 3D structure are displayed. For the complexes for which two different geometry can be found, both can be selected.  However, the complexes representations are qualitative and not quantitative. The goal is not to rely on the accuracy of the angles or bond length but to visualize the molecule in a global way. In addition, the package can provide either raman spectra, IR spectra or both depending on the user's preference. The width of the peaks can be selected and the transmittance style choosed. The precision of the peaks is also indicated. Finally, the data source is quoted at the bottom of the page.

Thus, our package is a coordination complex spectra predictor that can be used as an educational tool to obtain the main properties of the complex, its IR and Raman spectra and display the complex by the 2D and 3D representations.



## Installation

### Prerequisites

Before installing **CoordAnalyst**, make sure you have the following:

- Python 3.10 or higher
- [Anaconda](https://www.anaconda.com/download) or Miniconda (recommended)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/Carbone23400/CoordAnalyst.git
cd Project-Xplosion-Ba4
```

### Step 2 — Create a conda environment

We strongly recommend using a dedicated conda environment to avoid
dependency conflicts, particularly with RDKit.

```bash
conda create -n coordanalyst python=3.10
conda activate coordanalyst
```

### Step 3 — Install RDKit

RDKit must be installed via conda before the other dependencies,
as the pip version can cause conflicts with NumPy on some systems.

```bash
conda install -c conda-forge rdkit -y
```

### Step 4 — Install the package and its dependencies

```bash
pip install -e .
```

This installs CoordAnalyst in editable mode, meaning any changes
you make to the source code are immediately reflected without
needing to reinstall.

### Step 5 — Run the Streamlit app

```bash
python -m streamlit run App/streamlit_app.py
```

The app will open automatically in your browser at
`http://localhost:8501`.

---

### Dependencies

All Python dependencies are listed in `pyproject.toml` and installed
automatically in Step 4. The main ones are:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computations |
| `matplotlib` | Plotting |
| `plotly` | Interactive spectrum plots |
| `scipy` | Signal processing |
| `requests` | PubChem API calls |
| `streamlit` | Web interface |
| `pandas` | Data tables |
| `rdkit` | 2D molecular diagrams |
| `py3Dmol` | 3D molecular viewer |

---

### Troubleshooting

**`No module named 'coordchem'`**
Make sure you ran `pip install -e .` from the root of the repository
and that your conda environment is activated.

**`No module named 'rdkit'`**
RDKit must be installed via conda, not pip. Run:
```bash
conda install -c conda-forge rdkit -y
```
Then restart the app with `python -m streamlit run App/streamlit_app.py`
rather than `streamlit run`, to ensure the correct Python environment
is used.

**`No module named 'py3Dmol'`**
```bash
pip install py3Dmol
```

**Numpy version conflict with RDKit**
If you see errors mentioning `NumPy 1.x` and `NumPy 2.x`, downgrade
NumPy:
```bash
pip install "numpy<2"
```
Then reinstall RDKit via conda.




## examples of main functionality 










.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
