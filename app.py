# app.py
import os
import sqlite3
import datetime
import pickle
import numpy as np
import pandas as pd 
from flask import request, jsonify
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register a Unicode-capable font so the ₹ (Indian Rupee) symbol renders
# correctly in generated PDFs on every operating system, regardless of
# which fonts are installed locally.
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
try:
    pdfmetrics.registerFont(TTFont('DejaVuSans', os.path.join(_FONT_DIR, 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', os.path.join(_FONT_DIR, 'DejaVuSans-Bold.ttf')))
    PDF_FONT = 'DejaVuSans'
    PDF_FONT_BOLD = 'DejaVuSans-Bold'
except Exception:
    # Fallback: ₹ may not render correctly, but the app must not crash.
    PDF_FONT = 'Helvetica'
    PDF_FONT_BOLD = 'Helvetica-Bold'

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-only-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smartloan.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Allowed loan durations (in months) shown throughout the app.
VALID_LOAN_TERMS = [12, 24, 36, 48, 60, 84, 120, 180, 240, 360]
VALID_CREDIT_HIST = [0.0, 1.0]
VALID_PROPERTY_AREAS = ['Urban', 'Semiurban', 'Rural']


# ---------------------------------------------------------------------------
# Indian currency formatting helpers
# ---------------------------------------------------------------------------
def format_inr(value, decimals=0):
    """Format a number using the Indian numbering system, e.g. 1234567 -> 12,34,567"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value

    negative = value < 0
    value = abs(value)
    s = f"{value:.{decimals}f}"
    if decimals:
        integer_part, decimal_part = s.split(".")
    else:
        integer_part, decimal_part = s, ""

    if len(integer_part) <= 3:
        formatted = integer_part
    else:
        last3 = integer_part[-3:]
        rest = integer_part[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted = ",".join(groups) + "," + last3

    result = f"₹{formatted}"
    if decimals:
        result += f".{decimal_part}"
    return f"-{result}" if negative else result


app.jinja_env.filters['inr'] = lambda v: format_inr(v, 0)
app.jinja_env.filters['inr2'] = lambda v: format_inr(v, 2)


def calculate_emi(principal, annual_rate, months):
    """Standard reducing-balance EMI formula. Handles the zero-interest case safely."""
    principal = float(principal)
    annual_rate = float(annual_rate)
    months = int(months)

    if months <= 0:
        return 0.0, 0.0, 0.0

    if annual_rate == 0:
        emi = principal / months
    else:
        r = (annual_rate / 12) / 100
        emi = (principal * r * ((1 + r) ** months)) / (((1 + r) ** months) - 1)

    emi = round(emi, 2)
    total_payment = round(emi * months, 2)
    total_interest = round(total_payment - principal, 2)
    return emi, total_interest, total_payment


# --- Database Relational Architecture Maps ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    join_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class LoanPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    applicant_name = db.Column(db.String(150), nullable=False)
    gender = db.Column(db.String(20))
    married = db.Column(db.String(10))
    dependents = db.Column(db.String(10))
    education = db.Column(db.String(50))
    self_employed = db.Column(db.String(10))
    income = db.Column(db.Float)
    co_income = db.Column(db.Float)
    loan_amt = db.Column(db.Float)          # Full loan amount in ₹ (NOT thousands)
    term = db.Column(db.Integer)            # Loan duration in months
    interest_rate = db.Column(db.Float)     # Annual interest rate (%), user supplied
    credit_hist = db.Column(db.Float)
    property_area = db.Column(db.String(50))
    age = db.Column(db.Integer)

    # AI Output Engine Parameters
    loan_status = db.Column(db.String(10))
    approval_score = db.Column(db.Float)
    risk_level = db.Column(db.String(30))
    emi = db.Column(db.Float)
    total_interest = db.Column(db.Float)
    total_payment = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


# Safely load the pickled Random Forest Classifier Pipeline
try:
    with open('models/loan_model.pkl', 'rb') as f:
        ml_pipeline = pickle.load(f)
except FileNotFoundError:
    ml_pipeline = None


def ensure_schema():
    """Lightweight auto-migration: adds any columns missing from an older
    smartloan.db so upgrading this project never crashes on startup."""
    db_path = os.path.join('instance', 'smartloan.db')
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(loan_prediction)")
        existing_cols = {row[1] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        conn.close()
        return

    required_cols = {
        'interest_rate': 'FLOAT',
        'total_interest': 'FLOAT',
        'total_payment': 'FLOAT',
    }
    for col, col_type in required_cols.items():
        if col and col not in existing_cols:
            cur.execute(f"ALTER TABLE loan_prediction ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()


with app.app_context():
    db.create_all()
    ensure_schema()


# --- Security Session Guard Decorator ---
def login_required(f):
    import functools

    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Access Denied. Please authenticate your terminal to continue.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# ---------------------------------------------------------------------------
# Form validation helper for /predict
# ---------------------------------------------------------------------------
def validate_predict_form(form):
    """Returns (data_dict, errors_list). data_dict is only fully populated if
    errors_list is empty."""
    errors = []
    data = {}

    data['applicant_name'] = (form.get('applicant_name') or '').strip()
    if not data['applicant_name']:
        errors.append('Applicant name is required.')

    data['gender'] = form.get('gender')
    data['married'] = form.get('married')
    data['dependents'] = form.get('dependents')
    data['education'] = form.get('education')
    data['self_employed'] = form.get('self_employed')
    data['property_area'] = form.get('property_area')

    if data['property_area'] not in VALID_PROPERTY_AREAS:
        errors.append('Please select a valid property area.')

    def parse_float(field, label, min_value=None, allow_zero=True):
        raw = form.get(field, '').strip()
        if raw == '':
            errors.append(f'{label} is required.')
            return None
        try:
            val = float(raw)
        except ValueError:
            errors.append(f'{label} must be a valid number.')
            return None
        if val < 0:
            errors.append(f'{label} cannot be negative.')
            return None
        if not allow_zero and val == 0:
            errors.append(f'{label} must be greater than zero.')
            return None
        if min_value is not None and val < min_value:
            errors.append(f'{label} must be at least {min_value}.')
            return None
        return val

    data['age'] = parse_float('age', 'Age')
    if data['age'] is not None:
        data['age'] = int(data['age'])
        if not (18 <= data['age'] <= 75):
            errors.append('Age must be between 18 and 75.')

    data['income'] = parse_float('income', 'Applicant monthly income', allow_zero=False)
    data['co_income'] = parse_float('co_income', 'Co-applicant monthly income')
    data['loan_amt'] = parse_float('loan_amt', 'Loan amount', allow_zero=False)

    raw_term = form.get('term', '').strip()
    try:
        term_val = int(raw_term)
    except (TypeError, ValueError):
        term_val = None
        errors.append('Please select a valid loan duration.')
    if term_val is not None and term_val not in VALID_LOAN_TERMS:
        errors.append('Please select a valid loan duration.')
        term_val = None
    data['term'] = term_val

    data['interest_rate'] = parse_float('interest_rate', 'Interest rate')
    if data['interest_rate'] is not None and data['interest_rate'] > 50:
        errors.append('Interest rate looks unrealistic. Please enter a value under 50%.')

    raw_credit = form.get('credit_hist', '').strip()
    try:
        credit_val = float(raw_credit)
    except (TypeError, ValueError):
        credit_val = None
    if credit_val not in VALID_CREDIT_HIST:
        errors.append('Please select a valid credit history option.')
    data['credit_hist'] = credit_val

    if data['gender'] not in ('Male', 'Female'):
        errors.append('Please select a valid gender.')
    if data['married'] not in ('Yes', 'No'):
        errors.append('Please select a valid marital status.')
    if data['dependents'] not in ('0', '1', '2', '3+'):
        errors.append('Please select a valid number of dependents.')
    if data['education'] not in ('Graduate', 'Not Graduate'):
        errors.append('Please select a valid education level.')
    if data['self_employed'] not in ('Yes', 'No'):
        errors.append('Please select a valid self-employment status.')

    return data, errors


# --- Application Routing Core Engine ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''

        if not name or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Security mismatch. Passwords do not align.', 'danger')
            return redirect(url_for('register'))

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('This identity endpoint is already registered within our matrix.', 'warning')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='scrypt')
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration completed successfully! Proceed to sign-in terminal.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''

        # Admin Backdoor Access Parameter Layer
        if email == 'admin@smartloan.com' and password == 'Admin@2026':
            session['admin_logged_in'] = True
            session['user_name'] = 'System Administrator'
            session['user_id'] = 0
            flash('Administrative Core Environment Established.', 'success')
            return redirect(url_for('admin_dashboard'))

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            flash(f'Authentication verified. Access granted to node: {user.name}', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid identity parameters. Core authorization failed.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Terminal disconnected. Session tokens successfully cleared.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    predictions = LoanPrediction.query.filter_by(user_id=user_id).order_by(LoanPrediction.created_at.desc()).all()

    total = len(predictions)
    approved = len([p for p in predictions if p.loan_status == 'Approved'])
    rejected = total - approved
    app_rate = (approved / total * 100) if total > 0 else 0
    avg_loan_amt = (sum(p.loan_amt for p in predictions) / total) if total > 0 else 0

    return render_template(
        'dashboard.html',
        predictions=predictions,
        total=total,
        approved=approved,
        rejected=rejected,
        app_rate=app_rate,
        avg_loan_amt=avg_loan_amt,
    )


@app.route('/emi-calculator')
@login_required
def emi_calculator():
    return render_template('emi_calculator.html', valid_terms=VALID_LOAN_TERMS)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        data, errors = validate_predict_form(request.form)

        if errors:
            for err in errors:
                flash(err, 'danger')
            return redirect(url_for('predict'))

        emi_calc, total_interest, total_payment = calculate_emi(
            data['loan_amt'], data['interest_rate'], data['term']
        )

        # Processing AI Inference Vector.
        # The model was trained with LoanAmount expressed in thousands of ₹,
        # so we convert the user-facing rupee amount for the model only.
        loan_amt_for_model = data['loan_amt'] / 1000

        try:
            if ml_pipeline:
                input_df = pd.DataFrame([{
                    'Gender': data['gender'], 'Married': data['married'], 'Dependents': data['dependents'],
                    'Education': data['education'], 'Self_Employed': data['self_employed'],
                    'ApplicantIncome': data['income'], 'CoapplicantIncome': data['co_income'],
                    'LoanAmount': loan_amt_for_model, 'Loan_Amount_Term': data['term'],
                    'Credit_History': data['credit_hist'], 'Property_Area': data['property_area'],
                    'Applicant_Age': data['age']
                }])

                for col, le in ml_pipeline['encoders'].items():
                    input_df[col] = le.transform(input_df[col].astype(str))

                proba = ml_pipeline['model'].predict_proba(input_df)[0][1]
                status = 'Approved' if proba >= 0.5 else 'Rejected'
                score = round(proba * 100, 2)
            else:
                score = 80 if data['credit_hist'] == 1.0 and (data['income'] + data['co_income']) > (data['loan_amt'] / 10) else 20
                status = 'Approved' if score >= 50 else 'Rejected'
        except Exception:
            flash('The AI model could not process this application. Please check your inputs and try again.', 'danger')
            return redirect(url_for('predict'))

        risk = 'Low Risk' if score >= 75 else ('Medium Risk' if score >= 45 else 'High Risk')

        pred_record = LoanPrediction(
            user_id=session['user_id'], applicant_name=data['applicant_name'],
            gender=data['gender'], married=data['married'], dependents=data['dependents'],
            education=data['education'], self_employed=data['self_employed'],
            income=data['income'], co_income=data['co_income'],
            loan_amt=data['loan_amt'], term=data['term'], interest_rate=data['interest_rate'],
            credit_hist=data['credit_hist'], property_area=data['property_area'], age=data['age'],
            loan_status=status, approval_score=score, risk_level=risk,
            emi=emi_calc, total_interest=total_interest, total_payment=total_payment,
        )
        db.session.add(pred_record)
        db.session.commit()

        return redirect(url_for('result', id=pred_record.id))

    return render_template('predict.html', valid_terms=VALID_LOAN_TERMS)


@app.route('/result/<int:id>')
@login_required
def result(id):
    res = LoanPrediction.query.get_or_404(id)
    if res.user_id != session['user_id'] and not session.get('admin_logged_in'):
        flash('You are not authorized to view this record.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('result.html', res=res)


@app.route('/analytics')
@login_required
def analytics():
    user_id = session['user_id']
    preds = LoanPrediction.query.filter_by(user_id=user_id).all()

    app_count = len([p for p in preds if p.loan_status == 'Approved'])
    rej_count = len([p for p in preds if p.loan_status == 'Rejected'])
    avg_loan_amt = (sum(p.loan_amt for p in preds) / len(preds)) if preds else 0

    return render_template(
        'analytics.html', app_count=app_count, rej_count=rej_count,
        total=len(preds), avg_loan_amt=avg_loan_amt,
    )


@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        flash('Admin token absent. Re-authenticate credentials.', 'danger')
        return redirect(url_for('login'))
    users = User.query.filter(User.id > 0).all()
    preds = LoanPrediction.query.order_by(LoanPrediction.created_at.desc()).all()
    approved = len([p for p in preds if p.loan_status == 'Approved'])
    avg_loan_amt = (sum(p.loan_amt for p in preds) / len(preds)) if preds else 0
    return render_template(
        'admin.html', users=users, preds=preds,
        approved=approved, avg_loan_amt=avg_loan_amt,
    )


@app.route('/delete/prediction/<int:id>')
@login_required
def delete_prediction(id):
    pred = LoanPrediction.query.get_or_404(id)
    if session['user_id'] != pred.user_id and not session.get('admin_logged_in'):
        flash('Authorization violation.', 'danger')
        return redirect(url_for('dashboard'))

    db.session.delete(pred)
    db.session.commit()
    flash('Record removed successfully.', 'success')
    return redirect(url_for('admin_dashboard' if session.get('admin_logged_in') else 'dashboard'))


@app.route('/report/pdf/<int:id>')
@login_required
def download_pdf(id):
    res = LoanPrediction.query.get_or_404(id)
    if res.user_id != session['user_id'] and not session.get('admin_logged_in'):
        flash('You are not authorized to view this record.', 'danger')
        return redirect(url_for('dashboard'))

    os.makedirs('dataset', exist_ok=True)
    pdf_path = f"dataset/report_{id}.pdf"

    try:
        doc = SimpleDocTemplate(
            pdf_path, pagesize=letter,
            topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        )
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName=PDF_FONT_BOLD,
                                      fontSize=20, textColor=colors.HexColor('#4facfe'), spaceAfter=4)
        subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], fontName=PDF_FONT,
                                         fontSize=11, textColor=colors.HexColor('#555555'), spaceAfter=4)
        meta_style = ParagraphStyle('MetaStyle', parent=styles['Normal'], fontName=PDF_FONT,
                                     fontSize=9, textColor=colors.HexColor('#888888'), spaceAfter=14)
        section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontName=PDF_FONT_BOLD,
                                        fontSize=13, textColor=colors.HexColor('#111111'), spaceBefore=16, spaceAfter=8)
        body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontName=PDF_FONT, fontSize=10,
                                     spaceAfter=8, leading=14)

        story.append(Paragraph("SMART LOAN AI 2.0", title_style))
        story.append(Paragraph("Loan Approval Prediction Report", subtitle_style))
        story.append(Paragraph(
            f"Report generated on: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            meta_style
        ))

        status_color = colors.HexColor('#1e8e3e') if res.loan_status == 'Approved' else colors.HexColor('#d93025')
        decision_style = ParagraphStyle('DecisionStyle', parent=styles['Heading2'], fontName=PDF_FONT_BOLD,
                                         fontSize=16, textColor=status_color, spaceAfter=14)
        story.append(Paragraph(f"Loan Status: {res.loan_status.upper()}", decision_style))

        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fcfdfe')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ])

        story.append(Paragraph("Applicant Information", section_style))
        applicant_data = [
            [Paragraph("<b>Name</b>", body_style), Paragraph(res.applicant_name, body_style)],
            [Paragraph("<b>Age</b>", body_style), Paragraph(str(res.age), body_style)],
            [Paragraph("<b>Gender</b>", body_style), Paragraph(res.gender or '-', body_style)],
            [Paragraph("<b>Marital Status</b>", body_style), Paragraph(res.married or '-', body_style)],
            [Paragraph("<b>Education</b>", body_style), Paragraph(res.education or '-', body_style)],
            [Paragraph("<b>Employment Status</b>", body_style),
             Paragraph('Self-Employed' if res.self_employed == 'Yes' else 'Salaried', body_style)],
            [Paragraph("<b>Property Area</b>", body_style), Paragraph(res.property_area or '-', body_style)],
        ]
        t1 = Table(applicant_data, colWidths=[200, 300])
        t1.setStyle(table_style)
        story.append(t1)

        story.append(Paragraph("Financial Information", section_style))
        financial_data = [
            [Paragraph("<b>Applicant Monthly Income</b>", body_style), Paragraph(format_inr(res.income, 2), body_style)],
            [Paragraph("<b>Co-Applicant Monthly Income</b>", body_style), Paragraph(format_inr(res.co_income, 2), body_style)],
            [Paragraph("<b>Loan Amount</b>", body_style), Paragraph(format_inr(res.loan_amt, 0), body_style)],
            [Paragraph("<b>Interest Rate</b>", body_style), Paragraph(f"{res.interest_rate:.2f}% per annum", body_style)],
            [Paragraph("<b>Loan Duration</b>", body_style), Paragraph(f"{int(res.term)} Months", body_style)],
            [Paragraph("<b>Monthly EMI</b>", body_style), Paragraph(format_inr(res.emi, 2), body_style)],
            [Paragraph("<b>Total Interest Payable</b>", body_style), Paragraph(format_inr(res.total_interest, 2), body_style)],
            [Paragraph("<b>Total Payment</b>", body_style), Paragraph(format_inr(res.total_payment, 2), body_style)],
        ]
        t2 = Table(financial_data, colWidths=[200, 300])
        t2.setStyle(table_style)
        story.append(t2)

        story.append(Paragraph("Prediction Summary", section_style))
        prediction_data = [
            [Paragraph("<b>Loan Status</b>", body_style), Paragraph(res.loan_status, body_style)],
            [Paragraph("<b>Prediction Confidence</b>", body_style), Paragraph(f"{res.approval_score:.0f}%", body_style)],
            [Paragraph("<b>Risk Level</b>", body_style), Paragraph(res.risk_level, body_style)],
        ]
        t3 = Table(prediction_data, colWidths=[200, 300])
        t3.setStyle(table_style)
        story.append(t3)

        story.append(Spacer(1, 20))
        footer_style = ParagraphStyle('FooterStyle', parent=styles['Normal'], fontName=PDF_FONT,
                                       fontSize=8, textColor=colors.HexColor('#999999'))
        story.append(Paragraph(
            "This is a computer-generated academic decision-support report from SmartLoan AI 2.0 "
            "and does not represent a real bank's lending decision. Page 1.",
            footer_style
        ))

        doc.build(story)
    except Exception:
        flash('Could not generate the PDF report. Please try again.', 'danger')
        return redirect(url_for('result', id=id))

    return send_file(pdf_path, as_attachment=True, download_name=f"SmartLoanAI_Report_{id}.pdf")


# =========================
# SMARTLOAN AI CHATBOT
# =========================
@app.route("/chatbot", methods=["POST"])
def chatbot():
    if "user_id" not in session:
        return jsonify({"reply": "Please login first."})

    message = request.form.get("message", "").strip().lower()

    if not message:
        return jsonify({"reply": "Please type a message."})

    if any(word in message for word in ["hi", "hello", "hey", "good morning", "good evening"]):
        reply = """👋 Hello! Welcome to SmartLoan AI.

I'm your Smart Banking Assistant.

I can help you with:

• Loan Eligibility
• EMI Calculator
• Required Documents
• Credit Score
• Risk Analysis
• Approval Score
• Banking Guidance
• Loan Recommendations

How can I assist you today?"""

    elif "emi" in message:
        reply = """💰 EMI (Equated Monthly Installment)

EMI is the fixed amount paid every month until the loan is completely repaid.

EMI depends on:

• Loan Amount
• Interest Rate
• Loan Tenure

Use the SmartLoan AI EMI Calculator available in this system to estimate your monthly EMI instantly."""

    elif any(word in message for word in ["eligible", "eligibility", "loan approved", "loan rejected"]):
        reply = """🏦 Loan Eligibility

Our AI model analyzes:

✔ Income
✔ Loan Amount
✔ Credit History
✔ Employment
✔ Education
✔ Property Area

Based on these parameters SmartLoan AI predicts the loan approval possibility."""

    elif any(word in message for word in ["document", "documents", "aadhaar", "pan"]):
        reply = """📄 Required Documents

• Aadhaar Card
• PAN Card
• Passport Photo
• Address Proof
• Income Proof
• Salary Slips
• Bank Statement
• Income Tax Return (if applicable)"""

    elif any(word in message for word in ["credit", "cibil", "credit score", "credit history"]):
        reply = """📈 Credit Score

A CIBIL score above 750 improves approval chances.

Tips:

✔ Pay EMIs on time
✔ Don't miss credit card payments
✔ Avoid multiple loan applications
✔ Keep credit utilization low"""

    elif "risk" in message:
        reply = """⚠ Risk Levels

🟢 Low Risk

🟡 Medium Risk

🔴 High Risk

Risk depends upon income, repayment capacity, loan amount and credit history."""

    elif any(word in message for word in ["approval", "score", "chance"]):
        reply = """✅ Approval Score

Higher score means:

✔ Better financial profile
✔ Better repayment ability
✔ Higher loan approval probability"""

    elif "home loan" in message:
        reply = """🏠 Home Loan

Home loans help purchase or build a house.

Benefits:

✔ Long tenure
✔ Lower interest
✔ Tax benefits"""

    elif "car loan" in message:
        reply = """🚗 Car Loan

Banks generally finance up to 80–90% of the vehicle value depending on eligibility."""

    elif "personal loan" in message:
        reply = """💳 Personal Loan

Useful for:

• Medical Emergency
• Marriage
• Travel
• Education
• Home Renovation"""

    elif "education loan" in message:
        reply = """🎓 Education Loan

Education loans cover:

✔ Tuition Fees
✔ Hostel Fees
✔ Books
✔ Equipment"""

    elif "business loan" in message:
        reply = """🏢 Business Loan

Business loans help expand business, buy machinery and manage working capital."""

    elif "interest" in message:
        reply = """📊 Interest Rate

Interest depends on:

• Credit Score
• Income
• Loan Type
• Bank Policies
• Loan Tenure"""

    elif any(word in message for word in ["financial", "health", "finance"]):
        reply = """📈 Financial Health Tips

✔ Save regularly
✔ Control expenses
✔ Pay EMIs on time
✔ Avoid unnecessary debt
✔ Maintain an emergency fund"""

    elif any(word in message for word in ["recommend", "recommendation", "advice", "tips"]):
        reply = """💡 SmartLoan AI Recommendation

✔ Maintain good credit history
✔ Apply for an affordable loan amount
✔ Increase your monthly income
✔ Add a co-applicant
✔ Reduce existing liabilities"""

    elif any(word in message for word in ["thanks", "thank you"]):
        reply = """😊 You're Welcome!

Thank you for using SmartLoan AI.

Have a wonderful day!"""

    else:
        reply = """🤖 I couldn't understand your question.

You can ask about:

• EMI
• Loan Eligibility
• Credit Score
• CIBIL
• Documents
• Home Loan
• Car Loan
• Personal Loan
• Education Loan
• Business Loan
• Risk Analysis
• Approval Score
• Financial Health"""

    return jsonify({"reply": reply})


# --- Friendly error pages instead of raw tracebacks ---
@app.errorhandler(404)
def not_found_error(e):
    return render_template('error.html', code=404, message="The page you're looking for doesn't exist."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code=500, message="Something went wrong on our end. Please try again."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
