# yeastTLM
General machine learning pipeline for telomere maintenance analysis in fission and budding yeast.

## Requirements and Installation
Python 3.9.5+

Install requirements
```
pip3 install -r requirements.txt
```

## Usage
The following folders need to exist prior to running the code (similar structure to this repo): 

- data 
- features
- results
- tables
- figures

All, but the data folder will contain the outputs of the code excution. 
The data folder contains all of the data files needed.
> **Note:** The data folder contians large sized files (above 100 MB).

The final result should be:
```
.
├── constants.py
├── replicate_paper.py
├── models.py
├── features.py
├── helpers.py
├── figures
├── features
├── data
├── results
└── tables
```

Then, to replicate our Tables and Figures, run the following command:
```
python3 replicate_paper.py
```
