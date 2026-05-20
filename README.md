CoordAnalyst
========

## Description

### An educational tool helping students predict oxidation, geometry, number of d electrons, and other properties of a coordination complex as well as visualize it in 2D, 3D and draw a reliable IR and Raman spectra from streches in databases for common ligands.

The package CoordAnalyst aimes to analyse a wide range of coordination complexes. Indeed, the goal was to implement an educational tool to help better understand coordination chemistry by obtaining the basic informations and more. It meets a need for young chemists to have an easy-to-manipulate interface uniting all the properties of a complex they input.   
Thus, a user can enter the name or the formula of the complex on the interface to obtain its name and formula, the metal and its oxydation state, the geometry, the number of d electron and the coordination state. The list of the major common ligands that are supported is displayed, to help prevent any unknown input from the user. Furthermore, the 2D and 3D structure are displayed. For the complexes for which two different geometry can be found, both can be selected.  However, the complexes representations are qualitative and not quantitative. The goal is not to rely on the accuracy of the angles or bond length but to visualize the molecule in a global way. In addition, the package can provide Raman spectra, IR spectra or both depending on the user's preference. The width of the peaks can be selected and the transmittance style chosen. The precision of the peaks is also indicated. Finally, the data source is quoted at the bottom of the page.    
Thus, our package is a coordination complex spectra predictor that can be used as an educational tool to obtain the main properties of the complex, its IR and Raman spectra and display the complex by the 2D and 3D representations. It has value in being a lightweight tool, but its accuracy can not be expected to meet those of the result of quantum computations. 

    

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
