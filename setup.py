from setuptools import setup, find_packages

setup(
    name="blockgen",
    version="0.1",
    packages=find_packages(where="blockgen"),
    package_dir={"": "blockgen"},
)