from setuptools import setup , find_packages

setup(
    name = "stock_analysis",
    version='0.1',
    description='',
    author='Minseok Kwon',
    author_email="minsuk603@gmail.com",
    packages = find_packages,
    install_requires = [
        'numpy',
        'pandas',
        "yfinance",
        "matplotlib",
        "seaborn",
        "tensorflow",
        "scikit-learn",
        "pyarrow",
        "yahoo_fin"
    ]
)
