# TruthLens - Fake News Detection System

A Streamlit web application that analyses news article text with a trained machine learning model and estimates whether the content is likely REAL or FAKE.

## Features

- ✅ **English-only validation**: The app rejects non-English articles and prompts the user to paste English news only.
- ✅ **Real / Fake verdict**: Clear prediction badge and confidence score.
- ✅ **Detection signals**: A signal summary panel shows why the article was flagged.
- ✅ **Prediction logging**: Every analysis is saved automatically to a CSV log file.
- ✅ **Responsive UI**: Modern layout with mobile-friendly cards and signal panels.

## Installation

### 1. Clone the repository

```bash
cd fakeNewsDS
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r truthlens/requirements.txt
```

### 4. Run the app

```bash
streamlit run truthlens/truthlens_app.py
```

Open the app in your browser at `http://localhost:8501`.

## Usage

1. Paste the full news article text into the textarea.
2. Click **Analyse Article**.
3. If the text is valid and English, the model returns a verdict and confidence score.
4. Review the detection signals to understand the model's reasoning.
5. Clear the input with the **Clear** button and analyse the next article.

## How It Works

TruthLens evaluates article text using a trained classifier and TF-IDF features. It does not fact-check claims; instead, it analyses language, structure, and source-related signals that are statistically associated with misleading content.

### Input validation rules

- The article must contain at least 20 words.
- Only English news articles are accepted.
- Non-English input displays a user-facing error message.

## Prediction Logging

The app saves each prediction to `logs/prediction_logs.csv` in the repository root.

Each row includes:

- `timestamp`
- `article_text_snippet`
- `text_length`
- `prediction_label`
- `confidence_percent`
- `real_probability`
- `fake_probability`
- `model_version`
- `analysis_duration_seconds`

## Project Structure

```text
fakeNewsDS/
├── Models/
│   ├── trained_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metadata.json
├── Datasets/
├── logs/
│   └── prediction_logs.csv
├── truthlens/
│   ├── truthlens_app.py
│   ├── truthlens_detection_system.py
│   ├── style.css
│   ├── requirements.txt
│   └── README.md
└── metadata/
```

## Dependencies

The app installs dependencies from `truthlens/requirements.txt`, including:

- Streamlit
- scikit-learn
- joblib
- pandas
- numpy
- nltk
- langid

## Notes

- The model is a machine learning classifier, not a fact-checker.
- Predictions are probabilistic estimates and should be used for guidance only.
- The app is designed for educational and research use.

## Troubleshooting

### Model files missing

- Confirm `Models/trained_model.pkl` and `Models/tfidf_vectorizer.pkl` exist.
- Ensure the model folder is accessible from the repository root.

### English detection errors

- If your article is rejected, check that the text is English and contains enough sentences.
- Avoid pasting short fragments, headlines only, or text in another language.

### Log file issues

- The app creates `logs/prediction_logs.csv` automatically.
- Make sure the repository root has write permissions.

## License

For educational and research purposes only.
