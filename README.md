# ComEd Energy Consumption Prediction: A Time-Series Analysis

## 📌 Project Overview
Developed a predictive model to forecast hourly energy consumption for the Commonwealth Edison (ComEd) region. By integrating historical energy usage with local weather data, the project identifies key drivers of energy demand and provides accurate forecasts to assist in grid management and load balancing.

## 🧠 Technical Approach
This project involves a comprehensive end-to-end data science pipeline:
* **Data Integration:** Merged large-scale hourly energy consumption data with cleaned meteorological data (temperature, humidity, etc.).
* **Feature Engineering:** Extracted temporal features (hour, day of week, season) and lagged variables to capture time-series dependencies.
* **Modeling:** Experimented with multiple regression architectures, including **XGBoost** and **Random Forest**, to predict future energy loads.
* **Evaluation:** Assessed model performance using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

## 📊 Key Insights
* Identified a strong non-linear correlation between extreme temperatures and peak energy consumption.
* Achieved significant predictive accuracy, demonstrating the viability of using weather-augmented datasets for utility forecasting.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebooks

## 📂 Repository Structure
* `notebooks/`: Contains EDA and model training steps.
* `data/`: Processed datasets used for training.
* `reports/`: Detailed technical report and project proposal.
