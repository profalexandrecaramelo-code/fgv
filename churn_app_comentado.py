
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title="IA vs Intuição: Previsão de Churn", layout="wide")
st.title("IA vs Intuição: Previsão de Cancelamento de Clientes")

st.write("""
Este sistema permite comparar decisões humanas com previsões feitas por um modelo de IA. Faça upload de uma base de clientes e descubra quem está mais propenso a cancelar o serviço.
""")

uploaded_file = st.file_uploader("Faça upload do arquivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Informações Gerais da Base de Dados")
    st.markdown(f"- Total de clientes na base: **{df.shape[0]}**")
    st.markdown("""
    A tabela contém os seguintes campos:
    - `customerID`: Identificador único do cliente  
    - `gender`: Gênero do cliente  
    - `SeniorCitizen`: Se o cliente é idoso (1) ou não (0)  
    - `Partner`: Se o cliente tem parceiro(a)  
    - `Dependents`: Se possui dependentes  
    - `tenure`: Tempo de permanência (em meses)  
    - `PhoneService`: Se possui serviço telefônico  
    - `InternetService`: Tipo de serviço de internet  
    - `Contract`: Tipo de contrato (mensal, anual, etc.)  
    - `MonthlyCharges`: Valor mensal cobrado  
    - `TotalCharges`: Valor total cobrado  
    - `Churn`: Se o cliente cancelou (Yes) ou não (No)  
    """)

    st.subheader("🔍 Prévia dos dados (10 primeiros clientes)")
    st.dataframe(df.head(10))

    # Pré-processamento
    df_clean = df.copy()
    df_clean.drop('customerID', axis=1, inplace=True)
    df_clean.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)

    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col])

    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds_proba = model.predict_proba(X)[:, 1]
    df_result = df.copy()
    df_result['Risco_de_Churn'] = preds_proba
    df_result_sorted = df_result.sort_values(by='Risco_de_Churn', ascending=False)

    st.subheader("🚨 Top 5 Clientes com Maior Risco de Churn")
    st.dataframe(df_result_sorted[['customerID', 'Risco_de_Churn']].head(5))

    st.markdown("---")
    st.subheader("📈 Relatório de Classificação (modelo treinado com 80% dos dados)")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
