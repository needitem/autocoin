from setuptools import setup, find_packages

setup(
    name="autocoin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "requests",
        "python-dotenv",
        "redis",
        "PyJWT",
        "plotly",
    ],
    python_requires=">=3.8",
) 