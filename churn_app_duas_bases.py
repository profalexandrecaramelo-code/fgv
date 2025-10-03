
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title="AI in Churn prevention", layout="wide")
st.title("📉 AI to reduce customer disconnects")

st.write("""
You will need to upload 2 datasets:
1. One **labeled** with customers historical information (with field 'Churn') — for training and model testing.
2. Two **unlabeled** with customers that must be evaluated (without field 'Churn') — for risk inference.

This system will also suggest actions based on the patterns identified on customers that **never asked for a service disconnect**.
""")

# Upload training and test dataset
st.subheader("📁 1. Training and testing dataset (with 'Churn')")
train_file = st.file_uploader("Send the labeled dataset", type=["csv"], key="train")

# Upload evalution dataset
st.subheader("📁 2. Evaluation dataset (without 'Churn')")
predict_file = st.file_uploader("Send the customer dataset that we need to identify churn risks", type=["csv"], key="predict")

if train_file and predict_file:
    # Carregar e preparar base de treino
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
    st.dataframe(pd.DataFrame(report).transpose())

    # Process evaluation dataset
    st.markdown("---")
    st.subheader("🔎 Evaluation of new customers")

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
    st.write("Top 5 customers with higher risk of service disconnect:")
    st.dataframe(df_show.head(5))

    # Suggest actions on customers that did not request service disconnect
    st.markdown("---")
    st.subheader("💡 Recommendations to reduce Churn")

    df_retidos = df[df['Churn'] == 'No'].copy()
    comuns = df_retidos[['Contract', 'InternetService', 'tenure', 'MonthlyCharges']].mode().iloc[0]

    st.markdown("""
    Based on the list of customers that **never requested service disconnect**, please verifiy a recommendation:
    - 📌 **Type of contract** more stable: **{0}**
    - 🌐 **Prefered type of internet service**: **{1}**
    - ⏱ **Keep customers active for longer than** **{2} months**
    - 💰 **Ideal monthy recurring charge below ** **USD$ {3}**
    """.format(
        comuns['Contract'],
        comuns['InternetService'],
        int(comuns['tenure']),
        round(comuns['MonthlyCharges'], 2)
    ))
