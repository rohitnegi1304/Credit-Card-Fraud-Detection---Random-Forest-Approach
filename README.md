## Overview

This project focuses on implementing a Credit Card Fraud Detection system using machine learning techniques. The dataset used contains information about credit card transactions, and the goal is to build a model capable of identifying fraudulent transactions.

## Libraries Used

* **NumPy**: For numerical operations.
* **Pandas**: For data manipulation and analysis.
* **Scikit-learn:** For machine learning models and evaluation metrics.
* **TensorFlow and Keras**: For implementing the neural network.

## Dataset

The dataset used for this project is sourced from Credit Card Fraud Detection on [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](here). It consists of credit card transactions with features like amount, time, and anonymized variables.

## Data Preprocessing

* Checked for and handled null values.
* Balanced the dataset by oversampling the minority class (fraudulent transactions).
* Standardized the features using StandardScaler.
* Split the dataset into training and testing sets.

## Models Explored

1. **Logistic Regression**
1. **Linear Support Vector Classifier (SVC)**
1. **Random Forest Classifier**
1. **Multi-layer Perceptron (MLP) Classifier**
1. **Gaussian Naive Bayes**
1. **Decision Tree Classifier**
   
## Model Evaluation
Evaluated models on both training and testing sets:

* Logistic Regression:
    - Training Accuracy: 94.83%
    - Testing Accuracy: 93.97%
* Linear SVC:
    - Training Accuracy: 94.45%
    - Testing Accuracy: 95.48%
* Random Forest Classifier:
    - Training Accuracy: 100.0%
    - Testing Accuracy: 94.47%
* MLP Classifier:
    - Training Accuracy: 98.36%
    - Testing Accuracy: 94.97%
* Gaussian Naive Bayes:
    - Training Accuracy: 90.79%
    - Testing Accuracy: 90.95%
* Decision Tree Classifier:
    - Training Accuracy: 100.0%
    - Testing Accuracy: 91.46%
      
## Cross-Validation

Performed 5-fold cross-validation to assess model performance across different data splits:

* Logistic Regression: 92.54%
* Linear SVC: 61.79%
* Random Forest Classifier: 92.54%
* MLP Classifier: 49.6%
* Gaussian Naive Bayes: 84.67%
* Decision Tree Classifier: 90.72%
  
## Hyperparameter Tuning
Tuned hyperparameters for the Random Forest Classifier using GridSearchCV:

* Best Parameters: {'criterion': 'gini', 'n_estimators': 50}
* Best Score: 93.85%

## Neural Network Model
Implemented a neural network using TensorFlow and Keras:

* Architecture: Three hidden layers with ReLU activation, and output layer with sigmoid activation.
* Compiled with Adam optimizer and binary cross-entropy loss.

## Neural Network Performance

* Training Accuracy: 100%
* Testing Accuracy: 91.9%
  
## Final Model
Selected the Random Forest Classifier with parameters {'criterion': 'gini', 'n_estimators': 50} as the final model.

* Final Training Accuracy: 99.87%
* Final Testing Accuracy: 93.47%

## Conclusion

* The Random Forest Classifier outperformed the neural network slightly in terms of accuracy.
* The project provides a comprehensive exploration of various machine learning models for credit card fraud detection.
