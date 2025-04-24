
import streamlit as st
import numpy as np

st.set_page_config(page_title="Simulador de Perceptron", layout="centered")
st.title("🧠 Simulador Visual de Perceptron")
st.markdown("Este simulador demonstra como funciona um perceptron simples com duas entradas.")

# Entradas do usuário
st.subheader("1. Defina as entradas e pesos")
x1 = st.slider("Entrada x1 (ex: visitas no site)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
x2 = st.slider("Entrada x2 (ex: tempo na loja)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
w1 = st.slider("Peso w1", min_value=-2.0, max_value=2.0, value=0.4, step=0.1)
w2 = st.slider("Peso w2", min_value=-2.0, max_value=2.0, value=0.6, step=0.1)
bias = st.slider("Bias (b)", min_value=-5.0, max_value=5.0, value=-2.0, step=0.1)

# Cálculo da soma ponderada
z = x1 * w1 + x2 * w2 + bias

# Função de ativação: Step function
output = 1 if z > 0 else 0

# Exibir os resultados
st.subheader("2. Cálculo do perceptron")
st.write(f"Soma ponderada (z) = {x1} * {w1} + {x2} * {w2} + ({bias}) = **{z:.2f}**")
st.write(f"Saída após função de ativação (step): **{output}**")

# Explicação
st.markdown("""
### 🔍 Interpretação:
- Se **z > 0**, a saída do perceptron é **1** (ativado)
- Se **z ≤ 0**, a saída é **0** (não ativado)

Você pode usar este simulador para ensinar como o perceptron reage à variação de entradas, pesos e bias.
""")
