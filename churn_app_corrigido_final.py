
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title="IA para Redução de Churn", layout="wide")
st.title("📉 IA para Redução de Cancelamento de Clientes")

@st.cache_data
def carregar_dados(csv):
    return pd.read_csv(csv)

@st.cache_data
def preprocessar_dados(df, is_labeled=True):
    df = df.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)

    for col in df.select_dtypes(include='object').columns:
        if is_labeled and col == 'Churn':
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

@st.cache_resource
def treinar_modelo(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

st.subheader("📁 Passo 1: Base de Treino/Teste (com coluna 'Churn')")
train_file = st.file_uploader("Envie o arquivo CSV rotulado", type=["csv"], key="train")

modelo_treinado = None
if train_file:
    df = carregar_dados(train_file)
    st.success(f"Base carregada com {df.shape[0]} clientes.")

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
    st.dataframe(df.head(10))

    df_proc = preprocessar_dados(df)
    X = df_proc.drop('Churn', axis=1)
    y = df_proc['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("🚀 Treinar Modelo"):
        modelo_treinado = treinar_modelo(X_train, y_train)
        y_pred = modelo_treinado.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.subheader("📈 Relatório de Desempenho do Modelo")
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
        st.subheader("📁 Passo 2: Base para Avaliação (sem coluna 'Churn')")
        predict_file = st.file_uploader("Envie o arquivo com clientes a serem avaliados", type=["csv"], key="predict")

        if predict_file:
            df_novos = carregar_dados(predict_file)
            df_novos_proc = preprocessar_dados(df_novos, is_labeled=False)
            predicoes = modelo_treinado.predict_proba(df_novos_proc)[:, 1]
            df_novos['Risco_de_Churn'] = predicoes
            df_resultado = df_novos.sort_values(by='Risco_de_Churn', ascending=False)

            st.subheader("🔎 Top 5 clientes com maior risco de churn")
            st.dataframe(df_resultado.head(5))
