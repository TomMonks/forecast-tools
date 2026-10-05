# `forecast-tools`: fundamental tools to support the forecasting process in python.

[![DOI](https://zenodo.org/badge/250494795.svg)](https://zenodo.org/badge/latestdoi/250494795)
[![ORCID: Monks](https://img.shields.io/badge/ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)
[![PyPI version fury.io](https://badge.fury.io/py/forecast-tools.svg)](https://pypi.python.org/pypi/forecast-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomMonks/forecast-tools/master)
[![Python](https://img.shields.io/pypi/pyversions/forecast-tools)](https://pypi.org/project/forecasta-tools/)
[![Read the Docs](https://readthedocs.org/projects/pip/badge/?version=latest)](https://tommonks.github.io/forecast-tools/)

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

 forecast-tools has been developed to support forecasting education and applied forecasting research.  It is MIT licensed and freely available to practitioners, students and researchers via PyPi.  There is a long term plan to make forecast-tools available via conda-forge.

 ## Vision for forecast-tools

 1. Deliver high quality reliable code for forecasting education and practice with full documentation and unit testing.
 2. Provide a simple to use pythonic interface that users of `statsmodels` and `sklearn` will recognise.
 3. To improve the quality of Machine Learning time series forecasting and encourage the use of best practice.

## Features:

1. Implementation of classic naive forecast benchmarks such as Naive Forecast 1 along with prediction intervals
2. Implementation of scale-dependent, relative and scaled forecast errors.
3. Implementation of scale-dependent and relative metrics to evaluate forecast prediction intervals
4. Rolling forecast origin and sliding window for time series cross validation
5. Built in daily level datasets
6. An interactive plotting tool to visualise train test splits and forecasts.

## Ways to explore forecast-tools

1. `pip install forecast-tools`
2. Click on the launch-binder at the top of this readme. This will open example Jupyter notebooks in the cloud via Binder.
3. Read our [documentation on GitHub pages](https://tommonks.github.io/forecast-tools/)

## Citation

If you use forecast-tools for research, a practical report, education or any reason please include the following citation.

> Monks, Thomas. (2020). forecast-tools: fundamental tools to support the forecasting process in python. Zenodo. http://doi.org/10.5281/zenodo.3759863

```tex
@software{forecast_tools,
  author       = {Monks, Thomas},
  title        = {forecast-tools},
  month        = dec,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3759863},
  url          = {https://zenodo.org/doi/10.5281/zenodo.3759863}
}

```

## Contributing to forecast-tools

Please fork Dev, make your modifications, run the unit tests and submit a pull request for review.

> We provide a conda environment for development of forecast-tools. We recommend use of mamba as opposed to conda (although conda will work) as it is FOSS and faster.  Install from [mini-forge](https://github.com/conda-forge/miniforge)

Development environment:

```
mamba env create -f binder/environment.yml
mamba activate forecast_tools
```

Unit tests are provided and can be run via `hatch` and its coverage extension.  Run the following in the terminal.

To run tests in multiple Python environments (3.9-3.12)

```
hatch test --all
```

To obtain a coverage report run

```
hatch test --cover
```

**All contributions are welcome and must include unit tests!**

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://experts.exeter.ac.uk/19244-thomas-monks"><img src="https://avatars.githubusercontent.com/u/881493?v=4?s=100" width="100px;" alt="Tom Monks"/><br /><sub><b>Tom Monks</b></sub></a><br /><a href="https://github.com/TomMonks/forecast-tools/issues?q=author%3ATomMonks" title="Bug reports">🐛</a> <a href="https://github.com/TomMonks/forecast-tools/commits?author=TomMonks" title="Code">💻</a> <a href="#content-TomMonks" title="Content">🖋</a> <a href="#design-TomMonks" title="Design">🎨</a> <a href="https://github.com/TomMonks/forecast-tools/commits?author=TomMonks" title="Documentation">📖</a> <a href="#example-TomMonks" title="Examples">💡</a> <a href="#ideas-TomMonks" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-TomMonks" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-TomMonks" title="Maintenance">🚧</a> <a href="https://github.com/TomMonks/forecast-tools/commits?author=TomMonks" title="Tests">⚠️</a> <a href="#tutorial-TomMonks" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/amyheather"><img src="https://avatars.githubusercontent.com/u/92166537?v=4?s=100" width="100px;" alt="Amy Heather"/><br /><sub><b>Amy Heather</b></sub></a><br /><a href="https://github.com/TomMonks/forecast-tools/commits?author=amyheather" title="Documentation">📖</a> <a href="https://github.com/TomMonks/forecast-tools/issues?q=author%3Aamyheather" title="Bug reports">🐛</a> <a href="https://github.com/TomMonks/forecast-tools/commits?author=amyheather" title="Code">💻</a></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td align="center" size="13px" colspan="7">
        <img src="https://raw.githubusercontent.com/all-contributors/all-contributors-cli/1b8533af435da9854653492b1327a23a4dbd0a10/assets/logo-small.svg">
          <a href="https://all-contributors.js.org/docs/en/bot/usage">Add your contributions</a>
        </img>
      </td>
    </tr>
  </tfoot>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!