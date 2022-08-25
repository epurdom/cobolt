import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cobolt", # Replace with your own username
    version="0.0.1",
    author="boyinggong",
    author_email="boyinggong@berkeley.edu",
    description="A package for joint analysis of multimodal single-cell sequencing data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boyinggong/cobolt",
    project_urls={
        "Bug Tracker": "https://github.com/boyinggong/cobolt/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'umap-learn',
        'python-igraph',
        'sklearn',
        'xgboost',
        'pandas',
        'seaborn',
        'leidenalg'
    ],
    packages=setuptools.find_packages(exclude=['cobolt.tests']),
    python_requires=">=3.7",
)
