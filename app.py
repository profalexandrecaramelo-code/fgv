import streamlit as st
import numpy as np

# --- Funções da Rede Neural ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

# --- Dados de treinamento ---
entradas = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

saidas = np.array([[0], [1], [1], [0]])  # Problema XOR

# Inicialização dos pesos
np.random.seed(42)
pesos0 = 2 * np.random.random((2, 3)) - 1
pesos1 = 2 * np.random.random((3, 1)) - 1

# --- Interface Streamlit ---
st.title("Rede Neural Simples - XOR")
epocas = st.slider("Número de épocas de treinamento", 100, 10000, 5000, step=100)
taxa_aprendizado = st.slider("Taxa de aprendizado", 0.01, 1.0, 0.1, step=0.01)

# --- Treinamento ---
erros = []
for i in range(epocas):
    camada0 = entradas
    camada1 = sigmoid(np.dot(camada0, pesos0))
    camada2 = sigmoid(np.dot(camada1, pesos1))

    erro_camada2 = saidas - camada2
    media_erro = np.mean(np.abs(erro_camada2))
    erros.append(media_erro)

    derivada2 = erro_camada2 * sigmoid_derivada(camada2)
    derivada1 = derivada2.dot(pesos1.T) * sigmoid_derivada(camada1)

    pesos1 += camada1.T.dot(derivada2) * taxa_aprendizado
    pesos0 += camada0.T.dot(derivada1) * taxa_aprendizado

st.line_chart(erros, height=200, use_container_width=True)

st.subheader("Testar a rede")
x1 = st.selectbox("Entrada 1", [0, 1], key="x1")
x2 = st.selectbox("Entrada 2", [0, 1], key="x2")

entrada_usuario = np.array([[x1, x2]])
resultado = sigmoid(sigmoid(np.dot(entrada_usuario, pesos0)).dot(pesos1))

st.write(f"### Saída da rede: {resultado[0][0]:.4f}")
