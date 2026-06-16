# House Price Predictor

A machine learning project that predicts California housing prices using Linear Regression and Ridge Regression.

---

## Problem Statement

The objective of this project is to predict house prices based on housing-related features such as median income, house age, average rooms, population, latitude, and longitude. The project follows the complete machine learning workflow including EDA, preprocessing, model training, evaluation, and prediction.

---

## Dataset

California Housing Dataset from Scikit-learn

Rows: 20,640

Features: 8

Target: House Price

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

---# House Price Predictor

A Machine Learning project that predicts California housing prices using Linear Regression and Ridge Regression models.

---

## Problem Statement

The objective of this project is to predict house prices based on housing-related features such as median income, house age, average rooms, population, latitude, and longitude. The project follows a complete machine learning workflow including Exploratory Data Analysis (EDA), data preprocessing, model training, evaluation, and prediction.

---

## Dataset

Dataset: California Housing Dataset (Scikit-Learn)

* Number of records: 20,640
* Features: 8
* Target Variable: House Price

Features:

* MedInc (Median Income)
* HouseAge
* AveRooms
* AveBedrms
* Population
* AveOccup
* Latitude
* Longitude

Target:

* Price

---

## Project Structure

house-price-predictor/

├── data/

├── models/

├── notebooks/

│ └── eda.ipynb
      markdown

├── reports/

│ └── metrics.md
 
| Screenshots

├── src/

│ ├── preprocess.py

│ ├── train.py

│ ├── train_ridge.py

│ └── predict.py

├── .gitignore

├── README.md

└── requirements.txt

Running the Project

Data Preprocessing
python src/preprocess.py

Train Linear Regression Model
python src/train.py

Train Ridge Regression Model
python src/train_ridge.py

Predict House Price
python src/predict.py

## Usage Example

Input:

Median Income: 5

House Age: 25

Average Rooms: 6

Average Bedrooms: 1

Population: 1000

Average Occupancy: 3

Latitude: 37

Longitude: -122

Output:

Predicted House Price:
$2.87 (hundreds of thousands of dollars)

---

## Exploratory Data Analysis (EDA)

The following visualizations were performed:

1. House Price Distribution
2. Correlation Heatmap
3. Median Income vs House Price
4. Average Rooms vs House Price

Key Findings:

* Median Income has the strongest positive correlation with house price.
* Most house prices are concentrated between 1 and 3.
* Average rooms show a weaker relationship with house prices.

---

## Model Performance

| Model             | MAE    | RMSE   | R²     |
| ----------------- | ------ | ------ | ------ |
| Linear Regression | 0.5332 | 0.7456 | 0.5758 |
| Ridge Regression  | 0.5332 | 0.7456 | 0.5758 |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib

---

## Known Limitations

* Linear Regression assumes a linear relationship between features and target.
* Model performance can be improved using advanced algorithms such as Random Forest or Gradient Boosting.
* No web-based user interface has been implemented.

---

## Future Improvements

* Streamlit Web Application
* Hyperparameter Tuning
* Random Forest Regression
* Feature Engineering
* Model Deployment

## Author 
Sudeep S M


