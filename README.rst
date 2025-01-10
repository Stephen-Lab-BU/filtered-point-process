=========================
filtered-point-process
=========================

|ProjectStatus| |VersionDev| |BuildStatus| |License| |Publication|

.. |ProjectStatus| image:: http://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
   :alt: project status

.. |VersionDev| image:: https://img.shields.io/badge/version-in%20development-lightgrey
   :alt: version in development

.. |BuildStatus| image:: https://github.com/fooof-tools/fooof/actions/workflows/build.yml/badge.svg
   :target: https://github.com/fooof-tools/fooof/actions/workflows/build.yml
   :alt: build status

.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: license

.. |Publication| image:: https://img.shields.io/badge/paper-bioRxiv-green.svg
   :target: https://doi.org/10.1101/2024.10.01.616132
   :alt: publication

Beta package associated with the paper 
"Filtered Point Processes Tractably Capture Rhythmic And Broadband Power Spectral Structure 
in Field-based Neural Recordings" (2025)

Patrick F. Bloniasz, Shohei Oyama, Emily P Stephen

Package Setup
=============

1) Install Required Packages
----------------------------
While the package is in development, we recommend downloading in the following way:

Install Mamba through Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mamba is a drop-in replacement for conda, but is faster and better at resolving dependency conflicts.

.. code-block:: bash

   conda install mamba -n base -c conda-forge

Create an isolated environment & install filtered-point-process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/Stephen-Lab-BU/filtered-point-process.git
   cd filtered-point-process
   mamba env create -f environment.yml
   mamba activate filtered-point-process
   python -m pip install git+https://github.com/Stephen-Lab-BU/filtered-point-process.git

Major Upcoming Changes
----------------------
The package is being refactored to have a set of tutorials and be built on a multivariate version 
of this package. Currently, the package is set up to do univariate models. 
The full multivariate build with tutorials will be released as a complete build on November 27th.
