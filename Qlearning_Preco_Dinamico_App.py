
import streamlit as st
import numpy as np
import pandas as pd

st.title("💰 Preço Dinâmico com Q-Learning (Simulação)")

st.markdown("""
## 📌 Contexto
Você é um **consultor de IA** contratado por um e-commerce para maximizar o lucro através de preços dinâmicos.  
Seu desafio: Treinar um agente Q-Learning que aprenda a escolher o melhor preço em diferentes cenários de demanda.
""")

# Parâmetros
states = ["Alta Demanda", "Média Demanda", "Baixa Demanda"]
actions = ["Preço Baixo", "Preço Médio", "Preço Alto"]
n_states = len(states)
n_actions = len(actions)

# Q-Table
Q = np.zeros((n_states, n_actions))

# Parâmetros do aluno
alpha = st.slider("Taxa de aprendizado (α)", 0.01, 1.0, 0.1)
gamma = st.slider("Fator de desconto (γ)", 0.01, 1.0, 0.9)
epsilon = st.slider("Taxa de exploração (ε)", 0.0, 1.0, 0.2)
episodios = st.slider("Número de episódios de treino", 100, 5000, 1000)

# Função de recompensa
def obter_recompensa(state, action):
    tabela_recompensa = {
        (0, 0): 8, (0, 1): 10, (0, 2): 12,  # Alta demanda
        (1, 0): 5, (1, 1): 8, (1, 2): 6,   # Média demanda
        (2, 0): 3, (2, 1): 4, (2, 2): 2    # Baixa demanda
    }
    return tabela_recompensa[(state, action)]

# Treino
for _ in range(episodios):
    state = np.random.randint(0, n_states)
    if np.random.rand() < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(Q[state])

    reward = obter_recompensa(state, action)
    next_state = np.random.randint(0, n_states)

    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

# Mostrar Q-Table
df_q = pd.DataFrame(Q, index=states, columns=actions)
st.subheader("Q-Table final após o treino")
st.dataframe(df_q.style.format("{:.2f}"))

st.markdown("""
## 🎯 Atividade
✅ Analise a política aprendida (Q-Table).  
✅ Explique qual preço o agente recomenda para cada nível de demanda e por quê.  
✅ Sugira como o e-commerce poderia usar isso no mundo real.
""")
