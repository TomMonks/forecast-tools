# Change log

## v0.3.0 UNRELEASED

### Changed

* ENV: Development environment upgraded to Python 3.11 and libraries as of July 2024.

### Added

* BUILD: `pyproject.toml` added in migration from setuptools to hatch for package build and test management.

### Removed

* BUILD: `setup.py`, `requirements.txt`, `MANIFEST.in` no longer needed with hatch build backend.

### Fixed

* TESTING: `np.NAN` -> `np.nan` and `np.Inf` -> `np.inf`. The original constants have been deprecated in `numpy`
* MODEL: auto_naive error message patch.

## v0.2.1
* PATCH: fixed typo in setup.py
* DOCS: added missing CITATION.cff

## v0.2.0
* ENV: Updated environment to python 3.11 and package versions to latest in Dec 2023
* PATCH: np.float and np.int were deprecated in numpy v1.20.  Updated `baseline` module to use native 'float' and 'int' instead.
* PATCH: `cross_validation_folds` now returns a list of tuples instead of jagged array.
* ENHANCEMENT: added `metrics.winkler_score` and `metrics.absolute_coverage_difference` 
* DOCS: updated documentation to include "coverage" metrics.

## v0.1.7
* A more informative error message when training data is too short for naive method.
* Minor fixes of outdated docstrings in forecast_tools.baseline
* Improvements in the handling of invalid parameters by `auto_naive`.

## v0.1.6
* MIT LICENCE file added to PyPi package (previously missing).
* forecast_tools.feature_engineering module introduced
* forecast_tools.feature_engineering.sliding_window()

## v0.1.5

* auto_naive
* increased unit-test coverage (> 90%)

bug fix: mase for seasonal naive

## v0.1.4

cross-validation
mase





