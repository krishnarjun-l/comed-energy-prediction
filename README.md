# ComEd Energy Consumption Forecasting — Hybrid CNN-LSTM

Short-term electricity load forecasting for the ComEd (Commonwealth Edison) region using a hybrid **CNN-LSTM** deep learning model in PyTorch. The model predicts hourly energy consumption from historical load data combined with weather features.

This was the final project for a Neural Networks and Deep Learning course.

## Overview

Accurate short-term load forecasting helps grid operators balance supply and demand, schedule generation efficiently, and reduce costs. This project builds an end-to-end pipeline that:

1. Cleans and prepares hourly energy consumption data for the ComEd region.
2. Merges it with weather data and handles missing values.
3. Trains a hybrid **CNN-LSTM** model — 1D convolutional layers extract local temporal patterns, which an LSTM then uses to model longer-range sequential dependencies.
4. Evaluates forecasting accuracy using MAE, MSE/RMSE, and related metrics.

The approach is inspired by published work on hybrid CNN-LSTM models for short-term household and regional load forecasting (see [References](#references)).

## Data

This repository contains the **code and reports only** — the datasets are not included because of their size. The pipeline uses two public sources:

1. **ComEd / PJM hourly energy consumption** — hourly load data for PJM Interconnection regions, including ComEd.
   - https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
   - The pipeline specifically uses `COMED_hourly.csv`.

2. **Weather data** — hourly weather observations for the same period and region, cleaned and merged with the energy data in the notebooks.

After obtaining the data, the notebooks generate the intermediate files (`weather_clean.csv`, `weather_final.csv`, `merged_energy_weather.csv`) used by the modeling notebook. Place the source CSVs alongside the notebooks (or update the file paths near the top of each notebook).

## Pipeline / notebooks

Run the notebooks in order:

```
notebooks/
├── 01_energy_data_prep.ipynb            # Load ComEd hourly data; handle DST duplicate-timestamp issues
├── 02_missing_values_imputation.ipynb   # Clean weather data and impute missing values
├── 03_energy_consumption_prediction.ipynb  # Build & train the CNN-LSTM model; evaluate
└── Final_Project.ipynb                  # Consolidated end-to-end version
```

## Model

- **Architecture:** Hybrid CNN-LSTM — `nn.Conv1d` feature extractor followed by an `nn.LSTM` and a fully connected output head (PyTorch).
- **Preprocessing:** feature scaling with `StandardScaler`, sequence windowing for time-series input.
- **Training:** Adam optimizer, MSE loss, mini-batch training via `DataLoader`.
- **Evaluation:** Mean Absolute Error (MAE), Mean Squared Error (MSE) / RMSE.

## Repository structure

```
.
├── notebooks/
│   ├── 01_energy_data_prep.ipynb
│   ├── 02_missing_values_imputation.ipynb
│   ├── 03_energy_consumption_prediction.ipynb
│   └── Final_Project.ipynb
├── reports/
│   ├── Project_Proposal.pdf
│   └── Final_Project_Report.pdf
├── requirements.txt
└── README.md
```

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/ComEd-Energy-Forecasting.git
   cd ComEd-Energy-Forecasting
   ```

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the datasets (see [Data](#data)) and place the CSVs where the notebooks expect them.

5. Launch Jupyter and run the notebooks in order:

   ```bash
   jupyter notebook
   ```

## Requirements

- Python 3.10+
- PyTorch
- NumPy, pandas, scikit-learn, matplotlib, seaborn, tqdm
- A GPU is helpful but not required for this dataset size.

## References

The hybrid CNN-LSTM approach draws on:

- *Hybrid CNN-LSTM Model for Short-Term Individual Household Load Forecasting.*
- Short-term load forecasting literature published in *Applied Energy* (Elsevier).

(Original papers are not redistributed here; please consult them through their publishers.)

## Author

Krishnarjun Lakshminarayanan — Neural Networks and Deep Learning, final project.
