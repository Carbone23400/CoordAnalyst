============
Contributing
============

Welcome to ``CoordAnalyst`` contributor's guide.

This document explains how to get started contributing to the project.
All contributors are expected to be **open, considerate, reasonable,
and respectful**.


Issue Reports
=============

If you experience bugs or issues with ``CoordAnalyst``, please open
an issue on the `issue tracker`_. Include:

- Your operating system and Python version
- Steps to reproduce the problem
- The full error message if applicable


Development Setup
=================

Create an environment
---------------------

We recommend using conda to avoid dependency conflicts with RDKit::

    conda create -n coordanalyst python=3.10
    conda activate coordanalyst
    conda install -c conda-forge rdkit -y

Clone the repository
--------------------

1. Fork the repository on GitHub.
2. Clone your fork locally::

    git clone https://github.com/YourLogin/Project-Xplosion-Ba4.git
    cd Project-Xplosion-Ba4

3. Install the package in editable mode::

    pip install -e ".[dev]"

4. Verify the tests pass::

    python -m pytest tests/ -v


Implement your changes
----------------------

1. Create a branch for your changes::

    git checkout -b my-feature

   Never work directly on the ``main`` branch.

2. Make your changes. Add docstrings to any new functions or classes.

3. Add yourself to ``AUTHORS.rst``.

4. Add unit tests for any new functionality.

5. Commit your changes::

    git add <MODIFIED FILES>
    git commit -m "feat: describe your change here"

6. Check that all tests still pass::

    python -m pytest tests/ -v


Adding New Ligands to the Database
===================================

To add a new ligand:

1. Add it to ``KNOWN_LIGANDS`` in ``src/coordchem/parser.py``::

    "XYZ": ("ligand name", charge, denticity, "donor_atom"),

2. Add band entries to ``SEED_BANDS`` in ``data/ir_ra_bands.py``
   following the existing format. Always cite the source
   (Nakamoto page number or primary literature reference).

3. Run the database tests::

    python -m pytest tests/test_ir_bands.py -v


Submit your contribution
------------------------

1. Push your branch to GitHub::

    git push -u origin my-feature

2. Open a Pull Request on GitHub and describe your changes.
   At least one teammate should review before merging into ``main``.


Troubleshooting
---------------

**RDKit import errors**
    Install via conda, not pip::

        conda install -c conda-forge rdkit -y

    Run the app with ``python -m streamlit run`` rather than
    ``streamlit run`` to ensure the correct environment is used.

**NumPy version conflict**
    Downgrade NumPy::

        pip install "numpy<2"

**No module named coordchem**
    Make sure you ran ``pip install -e .`` from the repo root
    with your conda environment activated.


.. _repository: https://github.com/your-username/Project-Xplosion-Ba4
.. _issue tracker: https://github.com/your-username/Project-Xplosion-Ba4/issues