from setuptools import setup, find_packages
import os

# Read version from __about__.py
about = {}
with open(os.path.join("src", "gfpeps", "__about__.py")) as f:
    exec(f.read(), about)

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gfpeps",
    version=about["__version__"],
    author="Qi Yang",
    author_email="qiyang@mail.ustc.edu.cn",
    description="Gaussian Fermionic Tensor Network Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/unknown/gfpeps",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pymanopt==2.0.0",
        "h5py",
        "bitarray",
        "hydra-core",
        "jax==0.4.26",
    ],
    extras_require={
        "dev": [
            "pytest",
            "coverage[toml]>=6.5",
            "mypy>=1.0.0",
        ],
    },
) 