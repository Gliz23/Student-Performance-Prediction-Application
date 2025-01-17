# Student Performance Prediction App 📚🎓

Welcome to the **Student Performance Prediction App**! This web app allows you to predict student exam scores based on a set of performance factors like motivation, internet access, family income, and much more.

## Table of Contents

1. [Introduction](#introduction)
2. [App Overview](#app-overview)
3. [Features](#features)
4. [How to Use](#how-to-use)
5. [Screenshots](#screenshots)
6. [Prediction Results](#prediction-results)
7. [Technologies Used](#technologies-used)
8. [License](#license)

## Introduction

This app leverages machine learning to predict student exam scores based on various features. By entering values for different factors such as hours studied, motivation level, and more, you will get a predicted score!

## App Overview

The app has an easy-to-use interface where users can adjust the sliders and dropdown menus for categorical and numerical features to predict exam scores.

### Screenshot 1: App Overview
![App Overview](demo/1.png)

---

## Features

- **Sidebar with Interactive Inputs**: Users can input values through dropdown menus for categorical data and sliders for numerical values.
- **Machine Learning Model**: The app uses a trained machine learning model to predict the student's exam score.
- **Real-time Prediction**: When the user hits the "Predict Exam Score" button, the app displays the result.

---

## How to Use

1. **Step 1**: Open the app, and the first thing you will see is the **input parameter sidebar**.
2. **Step 2**: Select the categorical features like Motivation Level, Internet Access, etc.
3. **Step 3**: Adjust the numerical feature sliders such as Hours Studied, Attendance, etc.
4. **Step 4**: Click the **"Predict Exam Score"** button to view the predicted score based on the entered data.

---

## Screenshots

### Screenshot 2: Categorical Feature List (Sidebar)
![Categorical Feature List 1](demo/2.png)

### Screenshot 3: Categorical Feature List (Sidebar)
![Categorical Feature List 2](demo/3.png)

### Screenshot 4: Numerical Feature List
![Numerical Feature List](demo/4.png)

### Screenshot 5: Categorical Dropdown Menu
![Categorical Dropdown](demo/5.png)

### Screenshot 6: Final Prediction Result
![Final Prediction](demo/6.png)

---

## Prediction Results

Once the user selects their desired inputs and clicks the "Predict Exam Score" button, the predicted score will appear. The result is presented in a clean, centered format with a clear label showing the predicted exam score.

---

## Technologies Used

- **Python**: The app is built with Python and utilizes Streamlit for the frontend.
- **Machine Learning**: Trained model using algorithms like regression or decision trees.
- **Pandas**: Data manipulation and feature encoding.
- **Pickle**: Model and scaler saved and loaded using Pickle.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
