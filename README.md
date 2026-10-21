# Smart Loan AI

An academic Flask + Machine Learning web application that predicts loan approval outcomes using a Random Forest model. It includes an EMI calculator, PDF reports, a dashboard, and analytics.

> **Disclaimer:** This is an educational decision-support project, not financial advice and not a real bank's lending decision. Predictions may be inaccurate or biased and should not be used as the sole basis for lending decisions.

## Features

- User registration and login
- Loan eligibility prediction using a Random Forest model
- EMI calculator
- PDF report generation
- Dashboard and analytics
- SQLite database

## Tech Stack

Python, Flask, Flask-SQLAlchemy, SQLite, scikit-learn, pandas, NumPy, ReportLab, HTML, CSS, JavaScript, Bootstrap.

## Project Structure

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
│   └── loan_data.csv
├── models/
├── fonts/
├── static/
│   ├── css/
│   └── js/
└── templates/
```

## Run Locally

Use Python 3.10 or 3.11 for compatibility with the pinned package versions.

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment:

   **Windows**
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. If needed, train/regenerate the model:

   ```bash
   python train_model.py
   ```

6. Start the application:

   ```bash
   python app.py
   ```

7. Open `http://127.0.0.1:5000` in your browser.

## Configuration and Security

For local development, the app uses a development-only fallback secret. Before deploying, set a strong secret key through the `SECRET_KEY` environment variable. Never commit real passwords, production secrets, or personal database files.

**Windows PowerShell**
```powershell
$env:SECRET_KEY = "replace-with-a-long-random-secret"
python app.py
```

## Notes

- The SQLite database is created locally by the app; the local database file is intentionally not included in this repository.
- The `models/` directory may contain a pre-trained model. Run `train_model.py` if you want to retrain it.
- The bundled DejaVu fonts help PDF reports display the Indian Rupee symbol.

## Author

Aditya — B.Tech AI, Invertis University, Bareilly.
