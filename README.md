# Titanic-Machine-Learning-from-Disaster
This repository contains my first end-to-end machine learning project, developed as part of the classic Kaggle Titanic competition. The objective is to predict the survival of passengers on the Titanic based on features such as class, gender, age, and more.

Project Overview
Data Exploration:

Inspected and visualized the dataset to understand variable distributions and missing values.

Analyzed categorical and numerical features for predictive potential.

Data Preprocessing:

Imputed missing values (median for Age, mode for Embarked).

Converted categorical variables (Sex, Embarked) into numerical representations.

Dropped columns with excessive missing data (Cabin) and irrelevant features (Ticket, Name).

Feature Selection:

Selected key features: Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.

Modeling:

Implemented a baseline Random Forest Classifier using Scikit-Learn.

Trained the model on the provided training data.

Generated predictions for the test dataset.

Submission:

Created a submission file in the required Kaggle format (PassengerId, Survived).

Results
Public score: 0.75598 (75.6% accuracy) on the Kaggle leaderboard with basic preprocessing and no feature engineering.

Next Steps
Feature engineering (e.g., extracting titles from names, family size).

Model tuning and experimentation with other algorithms.

Improved handling of missing data and outliers.

Technologies used:
Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn
