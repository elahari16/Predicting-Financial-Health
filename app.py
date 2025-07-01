from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load('xgb_model.pkl')

# Raw features expected in uploaded CSV (before feature engineering)
raw_features = [
    'Age', 'Gender', 'Marital Status', 'Number of Dependents', 'Household Size',
    'Education Level', 'Occupation', 'Years in Current Job', 'Income Level',
    'Credit Score', 'Number of Credit Inquiries', 'Housing Status',
    'City or Region of Residence', 'Previous Bankruptcy Status', 'Health Condition',
    'Family Health History', 'Marital History', 'Residency Stability',
    'Financial Stability of Parents', 'Average Monthly Expenses', 'Credit Card Usage',
    'Savings Rate', 'Number of Loans Taken', 'Mortgage Information',
    'Investment Accounts', 'Emergency Fund Status', 'Loan Delinquencies History',
    'Bank Account Activity', 'Tax Filing History', 'Utility Bills Payment History',
    'Number of Credit Cards Held', 'Job Loss', 'Divorce History',
    'Major Medical Emergency', 'Adoption History', 'Bankruptcy History',
    'Health-related Legal Claims', 'Domestic or International Relocation',
    'Local Unemployment Rate', 'Inflation Rate', 'Interest Rates',
    'Economic Sentiment', 'Risk Tolerance', 'Financial Planner Involvement',
    'Debt-to-Income Ratio', 'Life Insurance Adequacy', 'Long-term Financial Goals'
]

# Final features used by the model
feature_order = raw_features + ['Expense_to_Income', 'Loan_to_Income', 'Credit_Utilization']
risk_map = {0: "Healthy", 1: "Moderate Risk", 2: "High Risk"}

@app.route('/')
def index():
    return '''
        <h2>📊 Financial Risk Prediction App</h2>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <p>Upload a CSV file with customer data (raw features only):</p>
            <input type="file" name="file">
            <input type="submit" value="Upload & Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return "❌ No file uploaded. Please upload a CSV."

    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)

    # Validate required base features
    missing = set(raw_features) - set(df.columns)
    if missing:
        return f"❗ Uploaded file is missing required columns: {missing}"

    # Automatically create the derived features
    df['Expense_to_Income'] = df['Average Monthly Expenses'] / (df['Income Level'] + 1)
    df['Loan_to_Income'] = df['Number of Loans Taken'] / (df['Income Level'] + 1)
    df['Credit_Utilization'] = df['Credit Card Usage'] / (df['Credit Score'] + 1)

    # Reorder columns and make prediction
    X = df[feature_order]
    df['Predicted Risk'] = model.predict(X)
    df['Risk Label'] = df['Predicted Risk'].map(risk_map)

    # Save the output
    output_path = 'predicted_risks.csv'
    df.to_csv(output_path, index=False)

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
