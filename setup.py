from setuptools import setup, find_packages

setup(
    name="artl-torch",
    version="0.2.0",
    description=(
        "Outlier-robust neural network training via the "
        "Augmented and Regularized Trimmed Loss (ARTL)"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
