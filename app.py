from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models & encoders
status_model = joblib.load("claim_status_model.pkl")
status_encoder = joblib.load("claim_status_encoder.pkl")

denial_model = joblib.load("denial_reason_model_grouped.pkl")
denial_encoder = joblib.load("denial_reason_encoder_grouped.pkl")

label_encoders = joblib.load("label_encoders_status.pkl")

# Routes for pages
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/introduction')
def introduction():
    return render_template('introduction.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/ml')
def ml():
    return render_template('ml.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    message = ''
    download_link = None

    if request.method == 'POST':
        try:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                # Save uploaded file
                file_path = os.path.join('static/Results', uploaded_file.filename)
                uploaded_file.save(file_path)

                # Load CSV
                df = pd.read_csv(file_path)

                # Drop unused columns
                df_encoded = df.copy()
                drop_cols = ['claim_id', 'patient_id', 'provider_id', 'service_date', 'claim_submission_date']
                df_encoded.drop(columns=[col for col in drop_cols if col in df_encoded.columns], inplace=True)

                # Encode object columns
                for col in df_encoded.select_dtypes(include='object').columns:
                    le = label_encoders.get(col)
                    if le:
                        df_encoded[col] = df_encoded[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
                        le.classes_ = np.unique(np.append(le.classes_, df_encoded[col].unique()))
                        df_encoded[col] = le.transform(df_encoded[col])

                # Predict claim status
                df['predicted_status'] = status_encoder.inverse_transform(status_model.predict(df_encoded))

                # Predict denial reason for Denied claims
                denied_indices = df[df['predicted_status'] == 'Denied'].index
                if not denied_indices.empty:
                    df_encoded.loc[denied_indices, 'is_denied'] = 1
                    expected_cols = denial_model.get_booster().feature_names

                    for col in expected_cols:
                        if col not in df_encoded.columns:
                            df_encoded[col] = 0

                    df_encoded = df_encoded[expected_cols]

                    denial_preds = denial_model.predict(df_encoded.loc[denied_indices])
                    denial_reasons = denial_encoder.inverse_transform(denial_preds.astype(int))
                    df.loc[denied_indices, 'predicted_denial_reason'] = denial_reasons
                else:
                    df['predicted_denial_reason'] = None

                # Save results
                result_path = os.path.join('static/Results', 'predicted_bulk.csv')
                df.to_csv(result_path, index=False)

                message = '✅ Bulk Prediction Completed!'
                download_link = url_for('static', filename='Results/predicted_bulk.csv')

            else:
                message = '⚠️ Please upload a valid CSV file.'

        except Exception as e:
            message = f'❌ Error occurred: {str(e)}'

    return render_template('prediction.html', message=message, download_link=download_link)

@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')

if __name__ == '__main__':
    app.run(debug=True)
