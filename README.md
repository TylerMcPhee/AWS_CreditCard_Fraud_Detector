# Credit Card Fraud Detection with AWS SageMaker XGBoost

## Overview

This project implements a machine learning pipeline for detecting credit card fraud using AWS SageMaker's XGBoost algorithm. It covers data preprocessing, model training, deployment as a real-time endpoint, and inference via an AWS Lambda function exposed through API Gateway.

---

## Project Components

### 1. Data Preprocessing

- Load credit card transaction data (`creditcards.csv`).
- Shuffle data
- Address class imbalance with `RandomOverSampler`.
- Split data into training and testing sets.
- Save processed datasets as CSV files without headers.
- Upload CSV files to an S3 bucket for SageMaker access.

### 2. Model Training

- Use SageMaker's XGBoost built-in algorithm.
- Configure estimator with hyperparameters (e.g., max_depth, eta).
- Train model on the balanced training data stored in S3.
- Validate using test data.

### 3. Model Deployment

- Deploy the trained XGBoost model to a SageMaker real-time endpoint.
- Endpoint serves inference requests.

### 4. Inference via AWS Lambda & API Gateway

- Lambda function receives JSON input with 31 features.
- Transforms input into CSV format and calls SageMaker endpoint.
- Returns predicted class label (fraud/non-fraud).
- API Gateway exposes Lambda function as a REST API.

### Prerequisites

- AWS account with SageMaker, S3, Lambda, and API Gateway permissions
- Python 3.x environment with packages:
  - `boto3`
  - `pandas`
  - `scikit-learn`
  - `imbalanced-learn`
  - `sagemaker`

---

## Usage Example

Sample JSON payload sent to API Gateway:

{
  "x1": 77967.0,
  "x2": 1.1499,
  "x3": 0.0202,
  ...
  "x30": 1.0975
}

Response from API:
{
  "prediction": 1 
}
(meaning it detects fraud)
