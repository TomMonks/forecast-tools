import setuptools

# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="forecast-tools", 
    version="0.1.4",
    author="Thomas Monks",
    author_email="forecast_tools@gmail.com",
    license="The MIT License (MIT)",
    description="Tools to support the forecasting process in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomMonks/forecast-tools",
    packages=setuptools.find_packages(),
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