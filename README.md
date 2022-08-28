# yeastTLM
General machine learning pipeline for telomere maintenance analysis in fission and budding yeast.

## Requirements and Installation
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
- figures.

All, but the data folder could be empty. 

Then, to replicate our Tables and Figures, run the following command:
```
python3 replicate_paper.py
```
