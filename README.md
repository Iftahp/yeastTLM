# A comparative analysis of telomere length maintenance circuits in fission and budding yeast
Data and code for the [article](https://github.com/Iftahp/yeastTLM/#) by Iftah Peretz, Martin Kupiec and Roded Sharan are available in this repository.

The paper aims at proposing a general machine learning pipeline for telomere maintenance analysis in fission and budding yeast.
The code and data are provided as-is and comes with no warranties. We tried to provide detailed documentation within the code. Any feedback is welcomed.

## Requirements and Installation
The code ran and tested on Python 3.9.5+.

The minimum version of dependencies are listed in the `requirements.txt` file and in order to install them execute:
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

All, but the data folder could be empty and will contain the outputs of the code excution. 

The data folder contains **all** of the data files needed.
> **Note:** The data folder contians large sized files (above 100 MB).

The final project directory should look like this:
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

Then, to replicate our **Tables** and **Figures**, run the following command:
```
python3 replicate_paper.py
```
![image](https://user-images.githubusercontent.com/8279861/187547179-d2426259-4900-420a-af2d-74e7ef2eb5f2.png)
