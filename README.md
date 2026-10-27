
# Installation Instructions
2. **Clone the repository:**

1) clone repos
2) create new conda env and activate it with python >=3.12:
   a) 'conda create -n context_based_anomalous_flow_detector python=3.12'
   b) conda activate context_based_anomalous_flow_detector
2) OR use venv
   a) python3 -m venv .venv_context_based_anom
   b) source .venv_context_based_anom/bin/activate 
3) make requirements
#3) Udpate pip:
#   python -m pip install -U pip
#4) Install project in editable mode:
#   pip install -e .
#5) create and install the ipykernel for the notebook:
#   a) python -m pip install ipykernel
#   b) python -m ipykernel install --user --name context_based_anomalous_flow_detector --display-name "Python (context_based_anomalous_flow_detector)"
4) in the empty data folder download the nf-unswb-nb15-v3 dataset and place the file
NF-UNSW-NB15-v3.csv directly into data. Rename to NF-UNSW-NB15-v3_raw.csv


# Context_Based_Anomalous_Flow_Detector

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Test if batches of anomalouse flows can be detected using a Bert-style approach.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         context_based_anomalous_flow_detector and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── context_based_anomalous_flow_detector   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes context_based_anomalous_flow_detector a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

