# Movie Genre Classification Web Application

## Overview

This project is a Machine Learning based web application that predicts the genre of a movie from its plot description. The system is built using Natural Language Processing techniques and deployed with a Flask web framework.

The model is trained on the Kaggle Genre Classification Dataset containing over 54,000 movie records.

---

## Features

- Multi-class genre prediction
- Text preprocessing using TF-IDF vectorization
- Machine Learning model using Multinomial Naive Bayes
- Confidence score display
- Clean and responsive web interface
- Flask-based backend integration

---

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML/CSS

---

## Dataset

- Source: Kaggle – Genre Classification Dataset
- Total Samples: 54,214
- Format: Text-based movie plot descriptions

---

## Model Details

- Vectorization: TF-IDF (max_features=5000)
- Algorithm: Multinomial Naive Bayes
- Train-Test Split: 80-20
- Model Accuracy: 52.29%

This is a multi-class classification problem, making 52% a reasonable baseline performance using a traditional ML approach.

---

## Project Structure

