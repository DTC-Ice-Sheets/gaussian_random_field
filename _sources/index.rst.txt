.. DTC Python Template documentation master file, created by
   sphinx-quickstart on Tue Jan 21 10:28:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gaussian Random Field Module
============================

The ``gaussian_random_field`` module is a sub-component of the
`Digital Twin Component for Ice Sheets (DTC-IS) <https://dtc-ice-sheets.org/>`_, an ESA-funded, EO-driven digital twin
project.

The ``gaussian_random_field`` module fits a Gaussian random field to spatially distributed ice-sheet data,
bringing variables of different resolution and sampling onto a shared grid and carrying an estimate of
uncertainty with them. It was developed to provide hydropower insights for the Watson River catchment on the
Greenland ice sheet, where it harmonises climate-model surface runoff and a satellite record of supraglacial
lake cover onto one mesh.

For a more interactive way of exploring the system, please visit the digital twin's
`dashboard <https://dashboards.dtc-ice-sheets.org/>`_ and/or explore the `notebooks <https://colab.research.google.com/github/DTC-Ice-Sheets/dtc_is_notebooks/>`_.

The remainder of this documentation page provides an overview of the ``gaussian_random_field`` module and details
on how to get it running locally.

.. toctree::
   :maxdepth: 1

   Overview <overview>
   Quick Start <quick-start>
   API Reference <api>
