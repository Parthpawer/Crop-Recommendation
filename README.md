# Crop Recommendation System

Welcome to the Crop Recommendation System GitHub repository! 🌱

This project utilizes machine learning to predict the best suitable crop based on environmental factors such as nitrogen, phosphorus, potassium, pH, temperature, and soil type. The aim of this project is to assist individuals in the farming industry by providing accurate crop recommendations.

## Project Overview

- **Accuracy**: The machine learning model achieves an impressive accuracy of 97%.
- **Data**: Currently trained on data from various districts of Maharashtra.
- **Future Plans**: We plan to expand our dataset to cover all regions of India in the future.

## Installation Guide

Follow these steps to set up and run the project on your local machine:

### Prerequisites

- Python 3.6+
- Git

### Steps

1. **Clone the repository:**

2. **Set up a virtual environment:**
- ```
  python -m venv virtualenv
  ```
3. **Activate the virtual environment:**
- On Windows:
  ```
  .\virtualenv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source virtualenv/bin/activate
  ```

4. **Install dependencies:**
- ```pip install -r requirements.txt```
5. **Generate model files:**
Run the `model_building.ipynb` Jupyter notebook to generate `model.pkl` and `scaler.pkl` files.

6. **Run the main application:**

Now you're all set to use the Crop Recommendation System! Happy farming! 🌾
