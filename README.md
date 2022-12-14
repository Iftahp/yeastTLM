# A comparative analysis of telomere length maintenance circuits in fission and budding yeast 
Data and code for the [article](https://www.frontiersin.org/articles/10.3389/fgene.2022.1033113/abstract) by Iftah Peretz, Martin Kupiec and Roded Sharan are available in this repository.

The paper aims at proposing a general machine learning pipeline for telomere length maintenance (TLM) analysis in fission and budding yeast.<br/>
The code and data are provided as-is and comes with no warranties. We tried to provide detailed documentation within the code.<br/>
Any feedback is welcomed (or there are problems with executing the code) - see the paper for ways to contact us.

## Requirements and Installation
The code was ran and tested on Python 3.9.5+.

The dependencies are listed in the `requirements.txt` file and in order to install them execute:
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

All, but the data folder could be empty and will eventually contain the outputs of the code excution. 

The data folder contains **ALL** of the data files needed.
> **Note:** The data folder contians some large sized files (above 100 MB). We include them to allow for a standalone application.

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

If you found this beneficial for your study, please cite the following:
```
@ARTICLE{PeretzTLM2022,
  
AUTHOR={Peretz, Iftah and Kupiec, Martin and Sharan, Roded},   
	 
TITLE={A comparative analysis of telomere length maintenance circuits in fission and budding yeast},      
	
JOURNAL={Frontiers in Genetics},      
	
VOLUME={13},           
	
YEAR={2022},      
	  
URL={https://www.frontiersin.org/articles/10.3389/fgene.2022.1033113},       
	
DOI={10.3389/fgene.2022.1033113},      
	
ISSN={1664-8021}
}
```
