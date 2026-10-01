# Silent Risk

Silent Risk is a corporate burnout wellness system built on the HackerEarth burnout dataset.

## Files

- `model_training.py`: trains the RandomForestRegressor and saves `burnout_model.pkl`
- `app.py`: Flask API for burnout prediction
- `dashboard.py`: Streamlit manager dashboard with a What-If simulator
- `utils.py`: shared feature engineering, risk sieve, and relief plan logic

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python model_training.py
```

## Run Both Apps

Open two terminals in this project directory.

Terminal 1:

```bash
python app.py
```

Terminal 2:

```bash
streamlit run dashboard.py
```

## API Endpoints

- `GET /health`
- `GET /feature-importance`
- `POST /predict`

Example JSON payload for `/predict`:

```json
{
  "Designation": 3,
  "Resource Allocation": 8,
  "Mental Fatigue Score": 8.5,
  "Tenure_Days": 120,
  "Workload_Intensity": 2.0
}
```

