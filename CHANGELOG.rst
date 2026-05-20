=========
Changelog
=========

Version 0.1.0 (2026)
=====================

First release of CoordAnalyst — a Python package for predicting
IR and Raman spectra of coordination complexes.

New Features
------------

- Formula parser for coordination complexes (``coordchem.parser``)
  supporting monodentate and polydentate ligands, oxidation state
  calculation, and coordination number detection
- IUPAC name parser for resolving complex names to structured data
  (``coordchem.name``)
- Geometry prediction based on coordination number and d-electron
  count (``coordchem.geometry``)
- IR and Raman band database seeded from Nakamoto 6th ed. covering
  30+ ligands and 200+ band entries (``coordchem.database``)
- Spectrum predictor with backbonding corrections, coordination
  shifts, and symmetry selection rules (``coordchem.spectra.predictor``)
- Gaussian broadening renderer with absorbance and transmittance
  display modes (``coordchem.spectra.renderer``)
- Interactive 2D molecular diagram using RDKit
- Interactive 3D molecular viewer using py3Dmol
- Streamlit web application with sidebar controls, ligand details,
  and band assignment table
- 162 unit tests across parser, database, and predictor modules