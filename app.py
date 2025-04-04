
import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Dados fictícios
dados = pd.DataFrame({
    'valor': [100, 2000, 50, 9000, 70, 8000, 300, 7500, 120, 1800, 90, 9500, 60, 11000],
    'transacoes_dia': [3, 25, 1, 30, 2, 28, 4, 22, 2, 20, 1, 35, 3, 40],
    'pais': ['BR', 'RU', 'BR', 'CN', 'BR', 'RU', 'BR', 'RU', 'BR', 'US', 'BR', 'CN', 'BR', 'US'],
    'fraude': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

# Separando variáveis
X = dados[['valor', 'transacoes_dia', 'pais']]
y = dados['fraude']

# Pipeline com pré-processamento e SVM
preprocessador = ColumnTransformer(
    transformers=[('pais', OneHotEncoder(drop='first'), ['pais'])],
    remainder='passthrough'
)
pipeline = Pipeline([
    ('preprocessamento', preprocessador),
    ('classificador', SVC(kernel='rbf', C=1.0))
])

# Treinamento
pipeline.fit(X, y)

# Interface Streamlit
st.title("🔍 Detecção de Fraude com SVM")
st.write("Preveja se uma transação é fraudulenta com base em valor, transações diárias e país.")

# Entradas do usuário
valor = st.number_input("Valor da transação", min_value=1, max_value=20000, value=500)
transacoes = st.slider("Número de transações do dia", 1, 50, 10)
pais = st.selectbox("País de origem da transação", options=['BR', 'RU', 'CN', 'US'])

# Previsão
if st.button("🔎 Verificar"):
    nova_transacao = pd.DataFrame({
        'valor': [valor],
        'transacoes_dia': [transacoes],
        'pais': [pais]
    })

    resultado = pipeline.predict(nova_transacao)[0]

    if resultado == 1:
        st.error("🚨 Transação suspeita: possivelmente **fraudulenta**.")
    else:
        st.success("✅ Transação considerada **legítima**.")
