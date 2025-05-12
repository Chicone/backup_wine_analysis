from setuptools import setup, find_packages

setup(
    name="gcmswine",
    version="0.1.0",
    description="GC-MS wine analysis tools",
    author="Luis Gomez Camara",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "fastdtw",
        "dcor",
        "myst-parser",
    ],
    python_requires=">=3.7",
)