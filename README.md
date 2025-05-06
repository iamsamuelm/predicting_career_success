# Predicting Career Success: The Impact of Education, Skills, and Networking

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Description

Career success is influenced by multiple factors, including academic performance, skills, and networking. However, students and job seekers often struggle to identify which aspects of their education and career development will have the greatest impact on their future success. 

This project uses predictive analytics to explore how academic performance, internships, projects, and networking contribute to career success. Using a dataset of 5,000 student records—including university ranking, GPA, skills, and career outcomes—we develop models to predict:

- Career Satisfaction
- Number of Job Offers
- Starting Salary

The results offer actionable insights to help students, career advisors, and educators optimize educational and career development strategies.

## Dependencies

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

## Environment Setup

1. Clone the repository
   
```
git clone https://github.com/iamsamuelm/predicting_career_success.git
cd predicting_career_success
```

2. (Optional) Create and activaet a virtual environment

```
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows
```

3. Install dependcies

```
pip install -r requirements.txt
```

## Running the Data Processing Pipeline
1. Make sure education_career_success.csv is placed in the project directory.
2. Run the data processing script:

```
python features.py
```

This script

- Handles outliers
- Encodes categorical features
- Engineers new features
- Scales numerical features
- Selects top features for model training
- Exports processed data to `.csv`

## Evaluating Models

After data preprocessing, run the model training script:

```
python train.py
```

This will: 

- Train predictive models for both Career Satisfaction and Starting Salary
- Use train/test splits for evaluation
- Output metrics like MAE, RMSE, and R²
- Generate performance visualizations

## Reproducing Results

To reproduce all results from scratch: 

```
python features.py
python predict.py
```

Make sure: 

- `education_career_success.csv` is in the root directory
- Your environment includes all required packages
- You are using a consistent random seed (`random_state=42` is set for reproducibility)

## Visualizations

To get initial data visualizations, run:

```
python plots.py
```

This creates: 

- Correlation heatmaps
- Distributions of target variables
- Boxplots grouped by field of study
- Scatter plots for key features vs outcomes

## Key Takeaways

This project provides data-driven insights into which factors most influence early career outcomes. The results empower stakeholders to invest effort in areas with the highest long-term ROI—academically and professionally. 

** The code in these files has to be updated. These instructions are for when I do.**


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
