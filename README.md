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

\- Designed for real-time, per-transaction fraud scoring with millisecond-level inference latency





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





\## System Architecture



```mermaid

flowchart LR

&nbsp;   A\[Incoming Transaction] --> B\[Feature Engineering]

&nbsp;   B --> C\[Preprocessing Pipeline]

&nbsp;   C --> D\[Trained ML Model]

&nbsp;   D --> E\[Fraud Probability Score]

&nbsp;   E --> F\[Fraud / Non-Fraud Decision]



&nbsp;   subgraph Training Pipeline

&nbsp;       G\[Historical Transactions] --> H\[Data Cleaning \& Encoding]

&nbsp;       H --> I\[Imbalance Handling]

&nbsp;       I --> J\[Model Training \& Evaluation]

&nbsp;       J --> D

&nbsp;   end



