# Bank Marketing Predictive Modeling

This repository contains the implementation of a machine learning project aimed at predicting whether a customer subscribes to a term deposit based on various demographic and behavioral features. The project employs both Random Forest and Neural Network classifiers to predict term deposit subscription from the Bank Marketing dataset available at the UCI Machine Learning Repository.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Neural Network Classifier](#neural-network-classifier)
  - [Ensemble Model: Combining Tuned Random Forest and Tuned Neural Network](#ensemble-model-combining-random-forest-and-neural-network)

- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Code Repository](#code-repository)
- [Contributors](#contributors)

## Introduction
The goal of this project is to predict whether a customer subscribes to a term deposit based on features such as age, job, marital status, education, and other relevant information. Two machine learning models were employed:
- Random Forest Classifier
- Neural Network Classifier

The project includes several techniques to handle class imbalance, including SMOTE (Synthetic Minority Over-sampling Technique), class weighting, threshold optimization, and the use of regularization.

## Dataset
The dataset used in this project is the **Bank Marketing** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). It contains information about clients of a Portuguese banking institution and includes the following features:
- Age
- Job
- Marital Status
- Education
- Default status
- Housing loan
- Personal loan
- Contact communication type
- Month of contact
- Previous campaign outcomes
- Subscription to a term deposit (target variable)

## Models

### Random Forest Classifier
A Random Forest model was trained to predict term deposit subscription. Various techniques were applied, including:
- **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance.
- **Class Weights** to adjust the importance of the minority class.
- **Threshold Optimization** based on ROC and Precision-Recall curves for improved classification performance.
- **Hyperparameter Tuning** using Grid Search and Randomized Search.

### Neural Network Classifier
The Neural Network was implemented and tested using the following variations:
- **Basic Neural Network** trained on the original dataset.
- **Optimized Neural Network** with regularization and class weights.
- **SMOTE-Augmented Neural Network** for class balancing.
- **Threshold Optimization** for better precision and recall balance.
- **Focal Loss-based Neural Network** to focus on difficult-to-classify examples.
- **Hyperparameter Tuning** for improved performance.


### Ensemble Model: Combining Tuned Random Forest and Tuned Neural Network

An ensemble model was created using a stacking technique to combine predictions from the Random Forest and Neural Network models, improving overall performance.


- **Base Predictions**: Both models generated probability predictions for training and test datasets.
- **Stacked Dataset**: These predictions were combined into a new dataset for the meta-model.
- **Meta-Model**: A Logistic Regression model was trained on the stacked dataset for final predictions.
- **Evaluation**: The ensemble achieved better precision, recall, and accuracy compared to individual models, effectively handling imbalanced data.

## Tech Stack
- **Python**: The primary programming language for implementing machine learning models.
- **TensorFlow**: Used for implementing and training Neural Network models.
- **scikit-learn**: Used for Random Forest Classifier, hyperparameter tuning, and data preprocessing.
- **imblearn**: For SMOTE and other balancing techniques.
- **Matplotlib & Seaborn**: For visualization of results and performance metrics.

## Installation
To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/SachithPathiranage/Predict-Term-Deposit.git
pip install -r requirements.txt
