from distutils.core import setup
from distutils.extension import Extension
from itertools import dropwhile
import numpy as np
from os import path


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def get_install_requirements():
    return [
        "numpy",
        "torch",
    ]


def setup_package():
    setup(
        name="interdiff",
        packages=["interdiff"],
        version="0.1",
        maintainer="Sirui Xu",
        maintainer_email="siruixu2@illinois.edu",
        url="https://github.com/Sirui-Xu",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
        ],
        install_requires=get_install_requirements(),
    )


if __name__ == "__main__":
    setup_package()