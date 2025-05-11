# Predicting Career Success

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Description

Career success is shaped by a wide range of factors—from academic performance and hands-on experience to networking and personal development. Yet for students and job seekers, identifying which aspects of their educational journey most influence future outcomes remains an ongoing challenge.

This project applies machine learning and predictive analytics to investigate how academic performance, internships, extracurricular involvement, and field of study contribute to early career success. Leveraging a dataset of 5,000 student records—including GPA, university ranking, skill assessments, and career outcomes—we engineered a holistic Composite Career Success Score that incorporates:

* Starting Salary
* Number of Job Offers
* Career Satisfaction
* Time to First Promotion

Using this target, we developed and evaluated multiple models, including Lasso Regression, Gradient Boosting, and a PyTorch-based Neural Network. Although the models exhibited limited predictive power on an individual level (R² ≈ 0.00), their consistent low error rates (MAE ≈ 0.15) suggest value for identifying general trends across large student cohorts.

The findings provide actionable insights for students, advisors, and educators seeking to optimize career development strategies, allocate support resources more effectively, and understand the nuanced limitations of algorithmic forecasting in human-centered domains.

## Dependencies

This project requires the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `joblib`
- `loguru`
- `tqdm`
- `typer`
- `torch`
- `missingno`

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
- Exports processed data to `.csv`

## Evaluating Models

After data preprocessing, run the model training script:

```
python predict.py
```

This will: 

- Train predictive models
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
- Boxplots for various categorical variables
- Scatter plots for key features vs outcomes

## Key Takeaways

This project provides data-driven insights into which factors most influence early career outcomes. The results empower stakeholders to invest effort in areas with the highest long-term ROI—academically and professionally. 


## Project Organization

```
├── data
│   └── processed/                         <- Cleaned and transformed data for modeling
│
├── docs
│   ├── docs/                              <- MkDocs documentation source files
│   ├── mkdocs.yml                         <- Configuration file for documentation site
│   └── README.md                          <- Documentation landing page
│
├── Makefile                               <- Build automation file for reproducible workflows
│
├── models
│   ├── gradient_boosting_regressor.joblib <- Trained Gradient Boosting model
│   ├── lasso_regression_model.joblib      <- Trained Lasso model
│   └── pytorch_neural_network.pth         <- Trained PyTorch model
│
├── notebooks
│   └── sprint-2.ipynb                     <- Main notebook for EDA and modeling
│
├── pyproject.toml                         <- Project metadata and configuration
├── README.md                              <- Project overview and setup instructions
│
├── references/                            <- Data dictionaries and supplementary materials
│
├── reports
│   ├── figures/                           <- Visualizations used in reporting
│   └── The Complexity of Success.pdf      <- Final project report
│
├── requirements.txt                       <- List of required Python packages
│
├── venv/                                  <- Virtual environment directory (should be in .gitignore)
│
└── Week_10/                                <- Source code and project logic
    ├── __init__.py
    ├── config.py                          <- Configuration variables
    ├── dataset.py                         <- Data loading and preprocessing
    ├── features.py                        <- Feature engineering
    ├── modeling/                          <- Model training and prediction modules
    └── plots.py                           <- Visualization and figure generation
```

--------
