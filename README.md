# Logistic Regression from Scratch

This project implements **Logistic Regression** manually in Python without using high-level machine learning libraries (like Scikit-Learn). It demonstrates the mathematical foundations of binary classification, including the Sigmoid activation function, Log Loss (Cost Function), and Gradient Descent optimization.

## 📂 Project Overview

The repository consists of two distinct implementations targeting different datasets:

### 1. Diabetes Prediction (`diabeties_perdiction.py`)
* **Type:** Multivariate Logistic Regression (Multiple Input Features).
* **Goal:** Predict the probability of a patient having diabetes based on medical statistics (e.g., Glucose levels).
* **Key Features:**
    * **Model Persistence:** Checks for a saved model (`logistic_model.npz`). If found, it loads weights instantly; otherwise, it trains a new model and saves it.
    * **Visualization:** Plots the "Probability of Diabetes" against Glucose levels while holding other features at their average.
    * **Accuracy:** Calculates and prints the training accuracy percentage.
* **Hyperparameters:**
    * Iterations: `5000`
    * Learning Rate ($\alpha$): `0.000245`

### 2. Purchase Decision (`purchase_discount.py`)
* **Type:** Univariate Logistic Regression (Single Input Feature).
* **Goal:** Predict if a customer will buy a product based on the discount offered.
* **Key Features:**
    * **Interactive Prediction:** Asks the user to input a specific discount value via the console to predict a purchase result.
    * **Visualization:** Plots the raw data points and the resulting Sigmoid curve (S-curve).
* **Hyperparameters:**
    * Iterations: `3500`
    * Learning Rate ($\alpha$): `0.0201`

---

## 🛠️ Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib
