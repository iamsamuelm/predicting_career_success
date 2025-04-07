# Predicting Career Success: The Impact of Education, Skills, and Networking

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## 📌 Project Description

Career success is influenced by multiple factors, including academic performance, skills, and networking. However, students and job seekers often struggle to identify which aspects of their education and career development will have the greatest impact on their future success. 

This project uses predictive analytics to explore how academic performance, internships, projects, and networking contribute to career success. Using a dataset of 5,000 student records—including university ranking, GPA, skills, and career outcomes—we develop models to predict:

- 🎯 Career Satisfaction
- 💼 Number of Job Offers
- 💰 Starting Salary

The results offer actionable insights to help students, career advisors, and educators optimize educational and career development strategies.

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
│                         Week 10 and configuration for tools like black
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
└── Week 10   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Week 10 a Python module
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

## 📦 Dependencies

This project requires the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `tensorflow`
- `keras-tuner`

You can install them all using:

```bash
pip install -r requirements.txt
```


