from setuptools import setup, find_packages

setup(
    name="mysupertools",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "multiply=mysupertools.tool.multiplication_a_b:multiply",
        ],
    },
)
