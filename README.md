# 🌸 Iris Flower Classification

Classify Iris flowers (Setosa, Versicolor, Virginica) from sepal/petal measurements,
with a full Streamlit frontend.

## Files
- `Iris.csv` — dataset (150 rows, from Kaggle/UCI)
- `train_model.py` — loads data, does EDA, trains Logistic Regression / KNN / Decision Tree,
  picks the best one, evaluates it, and saves the trained model to disk
- `app.py` — Streamlit frontend: sliders for input, live prediction, confidence chart,
  scatter plot of where your flower sits, model comparison table, dataset explorer
- `requirements.txt` — dependencies
- Generated after training: `iris_model.pkl`, `iris_scaler.pkl`, `iris_label_encoder.pkl`,
  `iris_metrics.pkl`, plus EDA plots (`eda_pairplot.png`, `eda_correlation.png`,
  `eda_histograms.png`, `confusion_matrix.png`)

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (creates the .pkl files the app needs)
python train_model.py

# 3. Launch the frontend
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Model results (on this dataset)
- Logistic Regression, KNN, and Decision Tree are all trained and compared.
- Best model is auto-selected by test accuracy (Logistic Regression won at ~93% here,
  KNN was close behind — Versicolor/Virginica overlap slightly, which is the hard part
  of this dataset; Setosa is always perfectly separable).

## Notes
- If you retrain (`python train_model.py`) the model files get overwritten — just
  refresh the Streamlit page (or restart it) to pick up the new model.
