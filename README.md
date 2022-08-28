# yeastTLM
General machine learning pipeline for telomere maintenance analysis in fission and budding yeast.

## Requirements and Installation
Python 3.9.5+

Install requirements
```
pip3 install -r requirements.txt
```

## Usage
The following folders need to exist: 

- data 
- features
- results
- tables
- figures

All, but the data folder could be empty. The data folder contains all of the data files.

Then, to replicate our Tables and Figures, run the following command:
```
python3 replicate_paper.py
```
