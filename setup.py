from setuptools import setup, find_packages
from pathlib import Path

requirements = Path('requirements.txt').read_text().splitlines()


long_description = Path('README.md').read_text()

setup(
    name='Fimatix_Title_classifier',  # Replace with your project name
    version='0.1.0',
    packages=['src'],
    install_requires=requirements,
    description='A tool for classifying sections in styled PDF documents as titles or non-titles using Random Forest or MLP.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Edward Crossing',
    author_email='edwardcrossing1@gmail.com',
    url='https://github.com/EdCrossing/Fimatix_Title_Classifier.git',  # Replace with your repository URL
    python_requires='>=3.6',
)