from setuptools import find_packages, setup

setup(
    name="atbe",
    version="0.2.0",
    description="Audio Transformations Based Explanations",
    url="https://github.com/D3annyC/atbe",
    author="ChengHan Chung, Anna Aljanaki",
    author_email="ch.chung127@gmail.com",
    license="MIT License",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib",
        "seaborn",
        "numpy",
        "pants",
        "pyyaml",
        "scipy",
        "librosa",
        "soundfile",
        "muda",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    include_package_data=True,
    zip_safe=False,
)
