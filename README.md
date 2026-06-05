In this project I want to create my own flow-classifier. I want to use ip-octet encoding.
Idea: a batch of flows are trained with a bert-style transformer. I learn a cls token with
 contrastive learning that describes the batch. Now I want to compare this to another 
batch where one or more flows look odd in their combination and the cls token will therefore
not be similar to one that has been learned before.

The first step is to create a representation of the batch of flows. What is
a representation for one flow? What is a representation for an IP address?

Ok - I realize all this is really complex and I should start with something simpler, debuggable.
So - I will start with SINGLE Flows and learn a cls token for a single flow. This way I will be
able to detect software that is wrong, e.g. flows where an unusual port is used together with
a certain ip or protocol etc.

# Installation Instructions
2. **Clone the repository:**
1) clone repos
2) create new conda env and activate it
3) pip install -e .
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

