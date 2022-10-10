# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import sys
from pathlib import Path

import setuptools

requirements = [
    "ipykernel",
    "marshmallow",
    "marshmallow_enum",
    "marshmallow_union",
    "matplotlib",
    "networkx",
    "retworkx==0.11.0",
    "numpy",
    "python-box",
    "pyyaml",
    "scipy",
    "sortedcollections",
    "engineering_notation",
    "intervaltree",
    "jsonschema",
    "pycparser",
    "requests",
    "setuptools>=40.1.0",
    "wheel",
    "zhinst==22.8.0",
    "zhinst-toolkit==0.4.0",
]


if not hasattr(setuptools, "find_namespace_packages") or not inspect.ismethod(
    setuptools.find_namespace_packages
):
    print(
        "Your setuptools version:'{}' does not support PEP 420 "
        "(find_namespace_packages). Upgrade it to version >='40.1.0' and "
        "repeat install.".format(setuptools.__version__)
    )
    sys.exit(1)


this_directory = Path(__file__).parent
version_path = this_directory / "laboneq/VERSION.txt"
version = version_path.read_text().rstrip()

long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="laboneq",
    version=version,
    description="Zurich Instrument tools for quantum information science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhinst/laboneq",
    author="Zurich Instruments Development Team",
    author_email="info@zhinst.com",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    keywords="zhinst sdk quantum",
    packages=setuptools.find_namespace_packages(
        exclude=["ci*", "docs*", "examples*", "resources*", "test*"]
    ),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
)
