
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Título
st.title("🔍 Previsão de Valor de Compra com Rede Neural")
st.subheader("Case: E-commerce de Eletrônicos")

# Explicação do case
st.markdown("""
Você é gestor de um e-commerce que vende eletrônicos.  
Seu objetivo é prever **quanto um cliente gastará** com base em características anteriores de compra:

- Frequência de compras anteriores  
- Valor médio das compras passadas  
- Tempo desde a última compra (em dias)  
- Número de itens comprados  
- Engajamento em campanhas anteriores
""")

# Gerar dados fictícios
np.random.seed(42)
n_samples = 200
data = pd.DataFrame({
    "frequencia_compras": np.random.randint(1, 10, size=n_samples),
    "valor_medio": np.random.uniform(50, 1000, size=n_samples),
    "dias_ultima_compra": np.random.randint(1, 120, size=n_samples),
    "num_itens": np.random.randint(1, 5, size=n_samples),
    "engajamento": np.random.uniform(0, 1, size=n_samples)
})

# Variável alvo (valor gasto): função não linear com ruído
data["valor_gasto"] = (
    20 * data["frequencia_compras"]
    + 0.5 * data["valor_medio"]
    - 0.3 * data["dias_ultima_compra"]
    + 15 * data["num_itens"]
    + 200 * data["engajamento"]
    + np.random.normal(0, 30, size=n_samples)
)

# Interface de seleção
st.sidebar.header("🧠 Parâmetros da Rede Neural")
hidden_layers = st.sidebar.slider("Número de neurônios na camada oculta", 1, 20, 5)
activation = st.sidebar.selectbox("Função de ativação", ["relu", "tanh", "logistic"])
camadas = st.sidebar.selectbox("Número de camadas ocultas", [1, 2])

if camadas == 1:
    hidden_layer_sizes = (hidden_layers,)
else:
    hidden_layer_sizes = (hidden_layers, hidden_layers)

# Separar dados
X = data.drop("valor_gasto", axis=1)
y = data["valor_gasto"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Treinar rede neural
model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                     max_iter=1000, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Resultados
st.markdown("### 📈 Comparação entre valor real e previsto")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Valor Real (R$)")
ax.set_ylabel("Valor Previsto (R$)")
ax.set_title("Valores Reais vs Previstos")
st.pyplot(fig)

st.markdown(f"**Erro Médio Absoluto (MAE):** R$ {mae:.2f}")
st.markdown(f"**R² Score:** {r2:.2f}")

# Mostrar parte dos dados
with st.expander("🔍 Ver dados de exemplo"):
    st.write(data.head(10))
