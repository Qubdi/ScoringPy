from setuptools import setup, find_packages

# python setup.py sdist bdist_wheel (1st in terminal)
# pip install dist/ScoringPy-0.0.2.tar.gz
# pip install dist/ScoringPy-0.0.2-py3-none-any.whl
# pip install --force-reinstall dist/ScoringPy-0.0.2-py3-none-any.whl
# twine upload dist/* (2nd in terminal)


setup(
    name="ScoringPy",  # The name of your package
    version="1.1.6",  # The version of your package
    author="Anri Tvalabeishvili, Kristine Dzneladze, Akaki Benidze",  # Your name or organization
    author_email="QubdiSolutions@gmail.com",  # Your contact email
    description="ScoringPy is an open-source Python library designed to streamline the development and deployment of classical credit scorecards.",  # A short description of your package
    long_description=open("README.md").read(),  # Long description read from a file
    long_description_content_type="text/markdown",  # The format of the long description (usually 'text/markdown')
    url="https://github.com/Qubdi/ScoringPy",  # URL to your project's homepage (GitHub, etc.)
    packages=find_packages(),  # Automatically find and include all packages in the project
    install_requires=[
        # List your package dependencies here
        # Example: 'requests', 'numpy', etc.
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
    ],
    project_urls={
        'Documentation': 'https://scoringpy.readthedocs.io/en/latest',
        'HomePage': 'https://github.com/Qubdi/ScoringPy',
        'Tracker': 'https://github.com/Qubdi/ScoringPy/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",  # Specify the Python versions supported
        "License :: OSI Approved :: MIT License",  # License type
        "Operating System :: OS Independent",  # OS compatibility
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
)
