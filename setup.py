import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()
long_description = 'Something to come.'

setuptools.setup(
    name="pychchpd",
    version="0.0.1rc1",
    author="Reza Shoorangiz",
    author_email="reza.shoorangiz@nzbri.org",
    description="A port of CHCHPD package from R.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rezashr/pychchpd",
    project_urls={
        "Bug Tracker": "https://github.com/rezashr/pychchpd/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=['gspread',
                      'gspread-dataframe',
                      'pandas>1',
                      'pyjanitor',
                      'numpy',
                      'python-dateutil',
                      'tabulate',
                      'google-api-python-client'],

    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
