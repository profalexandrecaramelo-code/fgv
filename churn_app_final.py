
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title="IA na Prevenção de Churn", layout="wide")
st.title("📉 IA para Redução de Cancelamento de Clientes")

st.write("""
Faça upload da base **rotulada** com informações históricas de clientes (com campo 'Churn') — para treino e teste do modelo.

Após o treino, será possível inserir uma segunda base com clientes que devem ser avaliados (sem campo 'Churn') — para previsão de risco.

O sistema também sugerirá ações com base nos padrões dos clientes que **nunca cancelaram o serviço**.
""")

# Upload da base de treino/teste
st.subheader("📁 Base de Treino/Teste (com 'Churn')")
train_file = st.file_uploader("Envie a base de dados rotulada", type=["csv"], key="train")

if train_file:
    # Carregar e preparar base de treino
    df = pd.read_csv(train_file)
    st.success(f"Base carregada com {df.shape[0]} clientes.")

    st.subheader("📊 Informações da Base de Dados")
    st.markdown(f"- Total de clientes: **{df.shape[0]}**")
    st.markdown("""
    A base contém os seguintes campos:
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
    st.subheader("📈 Relatório de Desempenho do Modelo")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Recomendações com base em clientes que não cancelaram
    df_retidos = df[df['Churn'] == 'No'].copy()
    comuns = df_retidos[['Contract', 'InternetService', 'tenure', 'MonthlyCharges']].mode().iloc[0]

    st.markdown("---")
    st.subheader("💡 Recomendações para Reduzir o Churn")
    st.markdown("""
    Com base nos clientes que **nunca cancelaram**, recomendamos:
    - 📌 **Tipo de contrato mais estável**: **{0}**
    - 🌐 **Tipo de internet preferido**: **{1}**
    - ⏱ **Manter clientes ativos por mais de** **{2} meses**
    - 💰 **Cobrança mensal ideal abaixo de** **R$ {3}**
    """.format(
        comuns['Contract'],
        comuns['InternetService'],
        int(comuns['tenure']),
        round(comuns['MonthlyCharges'], 2)
    ))

    st.markdown("---")
    st.subheader("📁 Agora envie a base de novos clientes para avaliação (sem 'Churn')")
    predict_file = st.file_uploader("Envie a base de avaliação", type=["csv"], key="predict")

    if predict_file:
        df_new = pd.read_csv(predict_file)
        df_show = df_new.copy()
        df_eval = df_new.copy()
        df_eval.drop('customerID', axis=1, inplace=True, errors='ignore')
        df_eval.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
        for col in df_eval.select_dtypes(include='object').columns:
            df_eval[col] = LabelEncoder().fit_transform(df_eval[col])

        preds_proba = model.predict_proba(df_eval)[:, 1]
        df_show['Risco_de_Churn'] = preds_proba
        df_show = df_show.sort_values(by='Risco_de_Churn', ascending=False)

        st.subheader("🔎 Top 5 clientes com maior risco de churn")
        st.dataframe(df_show.head(5))
