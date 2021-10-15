import setuptools
from setuptools.command.egg_info import egg_info


class egg_info_ex(egg_info):
    """Includes license file into `.egg-info` folder."""

    def run(self):
        # don't duplicate license into `.egg-info` when building a distribution
        if not self.distribution.have_run.get('install', True):
            # `install` command is in progress, copy license
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)

        egg_info.run(self)

# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="forecast-tools",
    version="0.1.6",
    author="Thomas Monks",
    author_email="forecast_tools@gmail.com",
    license="The MIT License (MIT)",
    license_files=('LICENSE', ),
    description="Tools to support the forecasting process in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomMonks/forecast-tools",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"": ["forecast_tools/data/*.csv"]},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
    install_requires=requirements,
)
