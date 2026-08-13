# Food Delivery Time Prediction

## 📌 Project Overview

This project focuses on predicting food delivery time using Machine Learning regression techniques.

The target variable is `Time_taken_min`, which represents the delivery time in minutes.

The project explores how factors such as road distance, average speed, preparation time, traffic level, weather, and other order-related features influence food delivery time.

## 🎯 Problem Statement

Build a Machine Learning regression model that predicts the time required to deliver a food order based on order, restaurant, rider, traffic, weather, and distance-related features.

## 📊 Dataset Features

The dataset contains features such as:

* Order Hour
* Day of Week
* Weekend indicator
* Festival indicator
* Weather
* Pickup Zone
* Dropoff Zone
* Vehicle Type
* Rider Experience
* Rider Rating
* Restaurant Rating
* Cuisine Type
* Number of Items
* Restaurant Load
* Preparation Time
* Road Distance
* Traffic Level
* Number of Signals
* Average Speed
* Delivery Priority

### Target Variable

`Time_taken_min`

## 🔎 Exploratory Data Analysis

The following EDA steps were performed:

* Checked missing values
* Checked duplicate records
* Investigated numerical feature distributions
* Investigated potential outliers using boxplots and IQR
* Analyzed numerical features against the target using scatterplots
* Analyzed categorical features against the target using boxplots
* Examined correlations between numerical variables

### Key EDA Findings

* `Road_Distance_km` has a strong positive relationship with delivery time.
* `Average_Speed_kmph` has a strong negative relationship with delivery time.
* `Preparation_Time_Min` has a weaker positive relationship with delivery time.
* Higher traffic levels are associated with higher delivery times.
* Rain and Storm conditions generally have higher delivery times than Clear and Cloudy conditions.
* `Restaurant_Load` shows a relatively weak individual relationship with delivery time.
* `Delivery_Priority` shows a relatively weak individual relationship with delivery time.
* `Time_taken_min` has a maximum recorded value of 180 minutes, with many observations at 180 minutes. These values were retained as valid target observations.

## 🛠️ Data Preprocessing

The following preprocessing steps were performed:

1. Removed irrelevant columns such as `Order_ID` and raw `Order_Date`.
2. Separated features (`X`) and target (`y`).
3. Split the data into training and testing sets.
4. Identified numerical and categorical features.
5. Applied `StandardScaler` to numerical features.
6. Applied `OneHotEncoder` to categorical features.
7. Combined the scaled numerical and encoded categorical features.

### Processed Data Shape

* Training samples: 40,000
* Testing samples: 10,000
* Numerical features after selection: 11
* One-hot encoded categorical features: 48
* Final features: 59

## 🤖 Machine Learning Model

### K-Nearest Neighbors Regression

`KNeighborsRegressor` was used because the target variable is continuous.

The initial model was trained with:

`k = 5`

### Initial Results

| Metric   |         Score |
| -------- | ------------: |
| MAE      | 10.41 minutes |
| RMSE     | 13.93 minutes |
| R² Score |        0.8472 |

## 🔧 K Value Experiment

Different values of K were tested:

|  K |   MAE |  RMSE |    R² |
| -: | ----: | ----: | ----: |
|  1 | 14.78 | 20.21 | 0.678 |
|  3 | 11.22 | 15.25 | 0.817 |
|  5 | 10.41 | 13.93 | 0.847 |
|  7 | 10.12 | 13.51 | 0.856 |
|  9 | 10.02 | 13.36 | 0.859 |
| 11 |  9.98 | 13.29 | 0.861 |
| 15 |  9.95 | 13.30 | 0.861 |
| 20 |  9.99 | 13.36 | 0.859 |
| 25 | 10.04 | 13.46 | 0.857 |
| 30 | 10.11 | 13.56 | 0.855 |

The results show that model performance improves substantially as K increases from 1 to around 11–15, after which performance starts to decline.

Cross-validation will be used to select the final K without using the test set for model selection.

## 📈 Evaluation Metrics

The model is evaluated using:

* **MAE** — average absolute prediction error in minutes
* **RMSE** — penalizes larger prediction errors
* **R² Score** — proportion of variance explained by the model

## 🚀 Future Improvements

* Perform KNN hyperparameter tuning using cross-validation.
* Compare KNN with Linear Regression.
* Compare with Ridge and Lasso Regression.
* Try Decision Tree and Random Forest Regression.
* Perform feature selection.
* Analyze prediction errors.
* Build a deployment interface for delivery-time prediction.

## 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Git
* GitHub

## 👨‍💻 Project Status

**In Progress**

Current stage:

`EDA → Feature Preprocessing → KNN Regression → Hyperparameter Tuning`

Future work will focus on cross-validation and comparison with other regression algorithms.
