import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title="AI for Churn Prevention", layout="wide" )
st.title("📉 AI for Customer Churn Reduction")

st.write("""
Upload two datasets:
1. A **labeled** dataset with historical customer information (with 'Churn' field) — for model training and testing.
2. An **unlabeled** dataset with customers to be evaluated (without 'Churn' field) — for risk prediction.

The system will also suggest actions based on patterns from customers who **never canceled the service**.
""")

# Upload training/testing dataset
st.subheader("📁 1. Training/Testing Dataset (with 'Churn')")
train_file = st.file_uploader("Upload the labeled dataset", type=["csv"], key="train")

# Upload evaluation dataset
st.subheader("📁 2. Customer Dataset for Evaluation (without 'Churn')")
predict_file = st.file_uploader("Upload customer dataset to predict risk", type=["csv"], key="predict")

if train_file and predict_file:
    # Load and prepare training dataset
    df = pd.read_csv(train_file)
    st.success(f"Training dataset loaded with {df.shape[0]} customers.")

    df_clean = df.copy()
    df_clean.drop('customerID', axis=1, inplace=True, errors='ignore')
    df_clean.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
    for col in df_clean.select_dtypes(include='object').columns:
        if col != 'Churn':
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col])
    df_clean['Churn'] = LabelEncoder().fit_transform(df_clean['Churn'])

    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.subheader("📊 Model Performance Report")
    report = classification_report(y_test, y_pred, output_dict=True)

    # Process evaluation dataset
    st.markdown("---")
    st.subheader("🔎 New Customer Evaluation")

    df_new = pd.read_csv(predict_file)
    df_show = df_new.copy()
    df_eval = df_new.copy()
    df_eval.drop('customerID', axis=1, inplace=True, errors='ignore')
    df_eval.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
    for col in df_eval.select_dtypes(include='object').columns:
        df_eval[col] = LabelEncoder().fit_transform(df_eval[col])

    preds_proba = model.predict_proba(df_eval)[:, 1]
    df_show['Churn_Risk'] = preds_proba
    df_show = df_show.sort_values(by='Churn_Risk', ascending=False)
    st.write("Top 5 customers with highest churn risk:")
    st.dataframe(df_show.head(5))

    # Suggest actions based on customers who didn't churn
    st.markdown("---")
    st.subheader("💡 Recommendations to Reduce Churn")

    df_retained = df[df['Churn'] == 'No'].copy()
    common_patterns = df_retained[['Contract', 'InternetService', 'tenure', 'MonthlyCharges']].mode().iloc[0]

    st.markdown("""
    Based on customers who **never canceled**, we recommend:
    - 📌 **Most stable contract type**: **{0}**
    - 🌐 **Preferred internet service**: **{1}**
    - ⏱ **Keep customers active for more than** **{2} months**
    - 💰 **Ideal monthly charge below** **${3}**
    """.format(
        common_patterns['Contract'],
        common_patterns['InternetService'],
        int(common_patterns['tenure']),
        round(common_patterns['MonthlyCharges'], 2)
    ))
