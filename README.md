\# Real-Time Transaction Fraud Detection



\## Overview

This project implements an end-to-end machine learning pipeline for detecting fraudulent financial transactions.

It supports real-time, single-transaction inference and is designed with class imbalance and model interpretability in mind.



\## Key Features

\- Synthetic transaction data generation

\- Feature engineering with temporal and behavioral signals

\- Class imbalance handling using oversampling

\- Model training and evaluation (Logistic Regression, Random Forest)

\- Real-time single-record fraud prediction

\- Model interpretability via feature coefficients



\## Tech Stack

\- Python

\- Pandas, NumPy

\- Scikit-learn

\- imbalanced-learn



\## Project Structure

\- src/ – core pipeline (data loading, preprocessing, training, inference)

\- models/ – trained model artifacts

\- data/ – generated datasets

\- tests/ – basic sanity tests



\## How to Run

python -m src.train



