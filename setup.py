import setuptools

setuptools.setup(
    name="PerioNet",
    version="0.0.1",
    author="Jonas",
    author_email="jonas@valfridsson.net",
    description="Periodic NN",
    url="https://github.com/kex2019/Utilities",
    packages=["perionet"],
    install_requires=["numpy==1.14.2", "tensorflow==1.12"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
