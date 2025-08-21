
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.title("Análise de Crédito — Rede Neural SEM Camada Oculta")

# Upload do CSV
uploaded_file = st.file_uploader("Faça upload do arquivo de dados (.csv)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Por favor, envie o arquivo 'dados_credito.csv' para continuar.")
    st.stop()

# Separar entradas e saída
if "aprovado" not in df.columns:
    st.error("A base deve conter a coluna 'aprovado' (0 = negado, 1 = aprovado).")
    st.stop()

X = df.drop("aprovado", axis=1)
y = df["aprovado"]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None)

# Padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo SEM camada oculta (equivalente a logística)
model = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Avaliação
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

# Contagem de parâmetros do modelo
# Usa os pesos e vieses aprendidos (coefs_ e intercepts_)
num_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)

num_samples = len(df)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Acurácia (teste)", f"{acc*100:.2f}%")
col2.metric("Precisão (teste)", f"{prec*100:.2f}%")
col3.metric("Recall (teste)", f"{rec*100:.2f}%")
col4.metric("Parâmetros do modelo", f"{num_params:,}")
col5.metric("Amostras na base", f"{num_samples:,}")

st.markdown("---")
st.markdown("### Simulação de um novo cliente")

# Inputs do usuário (ajuste conforme as colunas da sua base)
# Aqui assumimos 5 features comuns; se a base tiver outras colunas, o usuário pode usar a área abaixo para digitar manualmente.
default_inputs = {
    "renda": ("Renda mensal (R$)", 1000, 15000, 5000),
    "tempo": ("Tempo de emprego (anos)", 0.0, 10.0, 3.0),
    "idade": ("Idade", 18, 65, 35),
    "dividas": ("Valor total de dívidas (R$)", 0, 15000, 2000),
    "score": ("Score de crédito", 300, 850, 600),
}

input_values = {}
for col in X.columns:
    if col in default_inputs:
        label, a, b, c = default_inputs[col]
        if isinstance(a, float) or isinstance(b, float) or isinstance(c, float):
            input_values[col] = st.slider(label, float(a), float(b), float(c))
        else:
            input_values[col] = st.slider(label, int(a), int(b), int(c))
    else:
        # Campo numérico genérico para colunas não previstas
        input_values[col] = st.number_input(f"{col}", value=float(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0.0))

# Preparar input
input_array = np.array([list(input_values[c] for c in X.columns)])
input_scaled = scaler.transform(input_array)

# Previsão
prob = model.predict_proba(input_scaled)[0][1]
pred = model.predict(input_scaled)[0]

st.subheader("Resultado da Análise do Novo Cliente")
st.metric("Probabilidade de Aprovação", f"{prob*100:.1f}%")
if pred == 1:
    st.success("Crédito Aprovado")
else:
    st.error("Crédito Negado")

st.caption("Obs.: Este app mostra Acurácia, Precisão, Recall (conjunto de teste) e a **quantidade de parâmetros** aprendidos. Use o número de parâmetros para estimar o volume de dados necessário segundo a heurística discutida em aula.")
