# Titanic Survival Prediction

## Overview
I implemented a **Random Forest Classifier** to predict passenger survival on the Titanic. The model was trained on the provided dataset and achieved an accuracy of **83%** on the test set.

### 🔴 [Live Demo: Try it here!](https://titanicsurvivorprediction-j7h2pcjefmcxiay4svzgiv.streamlit.app/)

## Basis of Prediction
The model predicts survival based on several historical factors that influenced who had access to lifeboats:

1.  **Socio-Economic Status (Passenger Class)**: 
    - 1st-class passengers had significantly better access to lifeboats and were prioritized during the evacuation.
2.  **Gender ("Women and Children First")**: 
    - Women had a much higher survival rate (approx. 74%) compared to men (approx. 19%).
3.  **Age**: 
    - Children were prioritized for rescue. Adults in their prime also had better physical ability to reach safety.
4.  **Family Size**: 
    - Passengers with small families (1-3 members) often survived together.
    - Large families struggled to stay together and move quickly, often resulting in lower survival rates.

## Implementation Details

### 1. Data Loading & Preprocessing
- **Source**: Loaded `dataset/Titanic-Dataset.csv`.
- **Cleaning**:
    - **Age**: Imputed missing values with the median age.
    - **Embarked**: Imputed missing values with the mode ('S').
    - **Fare**: Imputed missing values with the median fare.
- **Encoding**:
    - **Sex**: Mapped 'male' to 0 and 'female' to 1.
    - **Embarked**: Mapped 'S' to 0, 'C' to 1, 'Q' to 2.
- **Feature Selection**: Used `['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']`.

### 2. Model Training
- **Algorithm**: Random Forest Classifier.
- **Parameters**: `n_estimators=100` (100 trees), `random_state=42` (for reproducibility).
- **Split**: 80% Training, 20% Testing.

### 3. Results
- **Accuracy**: **0.83** (83% of predictions were correct).

**Classification Report**:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Died)** | 0.84 | 0.88 | 0.86 | 105 |
| **1 (Survived)** | 0.81 | 0.76 | 0.78 | 74 |

---

## Installation & Usage

### 1. Prerequisites
Ensure you have Python installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training the Model
Run the script to train the model and generate the `titanic_model.pkl` file:

```bash
python titanicmodel.py
```

### 3. Running the Streamlit App
Launch the interactive web application to make predictions:

```bash
streamlit run app.py
```

---

## Docker Support

You can also run the application using Docker.

### Build the Image
```bash
docker build -t titanic-model .
```

### Run the Container
```bash
docker run titanic-model
```
