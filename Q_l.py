
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configurações iniciais
st.set_page_config(page_title="Q-Learning - Gerente de Estoque", layout="centered")
st.title("🚀 Q-Learning aplicado à Gestão de Estoque")

st.markdown("""
Este é um exemplo de aplicação de **Q-Learning**, um algoritmo de **Aprendizado por Reforço**, para um problema de **decisão de reabastecimento de estoque**.
O objetivo do agente é maximizar o lucro ao longo do tempo.
""")

# Parâmetros ajustáveis
col1, col2 = st.columns(2)
with col1:
    alpha = st.slider("Taxa de Aprendizado (alpha)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.slider("Fator de Desconto (gamma)", 0.01, 1.0, 0.9, 0.01)
with col2:
    epsilon = st.slider("Exploração Inicial (epsilon)", 0.01, 1.0, 0.5, 0.01)
    n_episodes = st.slider("Número de Episódios (dias simulados)", 10, 500, 100, 10)

# Definindo ambiente
states = ['Baixo', 'Médio', 'Alto']
actions = ['Comprar', 'Nao Comprar']
q_table = np.zeros((len(states), len(actions)))

def demanda():
    return np.random.choice(['Baixa', 'Alta'], p=[0.4, 0.6])

def recompensa(estoque, acao, demanda):
    custo_fixo = 5
    preco_venda = 10
    custo_reabastecimento = 4
    if acao == 'Comprar':
        estoque += 1
        lucro = -custo_reabastecimento
    else:
        lucro = 0

    if demanda == 'Alta':
        venda = min(estoque, 2)
    else:
        venda = min(estoque, 1)

    lucro += venda * preco_venda - custo_fixo
    estoque -= venda
    return lucro, estoque

def estado_index(estoque):
    if estoque <= 1:
        return 0  # Baixo
    elif estoque == 2:
        return 1  # Médio
    else:
        return 2  # Alto

# Treinamento
estoque_hist = []
lucro_hist = []
estoque = 2

for episodio in range(n_episodes):
    s_idx = estado_index(estoque)
    if np.random.rand() < epsilon:
        a_idx = np.random.choice(len(actions))
    else:
        a_idx = np.argmax(q_table[s_idx])

    acao = actions[a_idx]
    dem = demanda()
    r, novo_estoque = recompensa(estoque, acao, dem)
    s_prime_idx = estado_index(novo_estoque)

    # Atualização Q-Learning
    q_table[s_idx, a_idx] += alpha * (r + gamma * np.max(q_table[s_prime_idx]) - q_table[s_idx, a_idx])

    estoque = novo_estoque
    estoque_hist.append(estoque)
    lucro_hist.append(r)

# Resultados
st.subheader("Q-Table Final")
q_df = pd.DataFrame(q_table, index=states, columns=actions)
st.dataframe(q_df.style.highlight_max(axis=1))

st.subheader("Lucro acumulado por episódio")
cumulative = np.cumsum(lucro_hist)
fig, ax = plt.subplots()
ax.plot(cumulative)
ax.set_xlabel("Episódio")
ax.set_ylabel("Lucro acumulado")
ax.grid(True)
st.pyplot(fig)

st.subheader("Histórico de Estoque")
fig2, ax2 = plt.subplots()
ax2.plot(estoque_hist, color='orange')
ax2.set_xlabel("Episódio")
ax2.set_ylabel("Nível de Estoque")
ax2.set_yticks([0, 1, 2, 3])
ax2.grid(True)
st.pyplot(fig2)

st.success("Simulação concluída! Ajuste os parâmetros e rode novamente para comparar os resultados.")
