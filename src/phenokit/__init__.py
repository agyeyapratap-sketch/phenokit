"""phenokit — sensor-to-insight toolkit for field crop phenotyping.

Covers the full chain used on handheld, ground-based and aerial (UAV) phenotyping
platforms:

* ``acquisition``  – platform/sensor specs, flight & transect planning, solar windows,
  acquisition-protocol checklists.
* ``catalog``      – FAIR/MIAPPE-style metadata, standard directory layout, SHA-256
  manifests, SQLite data catalogue with provenance.
* ``qc``           – integrity and quality checks for RGB, spectral (point & imaging),
  thermal, LiDAR and annotation data.
* ``calibration``  – empirical-line & dark/white reflectance calibration, spectroradiometer
  splice correction, thermal emissivity correction.
* ``spectral`` / ``indices`` – preprocessing and a vegetation-index library.
* ``segmentation`` / ``plots`` / ``pointcloud`` / ``thermal`` – computer-vision and
  plot-level trait extraction.
* ``ml`` / ``deep`` / ``validation`` – leakage-aware machine learning, deep models,
  agreement statistics, heritability and cross-species/cross-platform validation.
* ``datasets`` – verified downloaders for public tutorial datasets.
* ``synthetic`` – a multi-crop, multi-platform synthetic trial with known ground truth.
"""

__version__ = "0.1.0"
