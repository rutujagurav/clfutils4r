import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clfutils4r", # Replace with your own username
    version="0.0.3",
    author="Rutuja Gurav",
    author_email="rutujagurav100@gmail.com",
    description="Wrapper around some basic sklearn and scikit-plot utilities for classification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rutujagurav/clfutils4r",
    project_urls={
        "Bug Tracker": "https://github.com/rutujagurav/clfutils4r/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'deprecation', 'seaborn', 'scikit-learn', 'scikit-plot', 'shap'
      ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)