
import streamlit as st
import numpy as np
import pandas as pd

st.title("💰 Preço Dinâmico com Q-Learning — Simulação no E-commerce")

st.markdown("""
## 📌 Case: Otimização de Preço Dinâmico no E-commerce

Você é um **consultor de IA** contratado por um e-commerce.  
Esse e-commerce quer **maximizar o lucro** ajustando dinamicamente o preço de um produto ao longo do tempo, em diferentes situações de demanda.

➡ **Desafio:** Descobrir qual preço aplicar em cada cenário de demanda (alta, média, baixa) para obter o maior lucro possível.  
➡ **Solução proposta:** Usar o **Q-Learning** para o agente aprender, por tentativa e erro, qual preço escolher em cada situação.

💡 **No final, o aluno deve interpretar a Q-table e explicar como o e-commerce poderia usar essa política para definir seus preços reais.**
""")

st.markdown("""
### 🔧 Hiperparâmetros do Q-Learning
- **Taxa de aprendizado (α)**: controla o quanto o agente aprende com experiências novas.  
  - Valor alto: aprende rápido, mas esquece rápido.  
  - Valor baixo: aprende devagar, mas de forma estável.
- **Fator de desconto (γ)**: define o peso das recompensas futuras.  
  - Valor alto: pensa no longo prazo.  
  - Valor baixo: valoriza o ganho imediato.
- **Taxa de exploração (ε)**: controla o quanto o agente explora ações novas ao invés de usar o que já sabe.  
  - Valor alto: explora mais no começo.
- **Número de episódios**: quanto mais episódios, mais o agente tem chance de aprender.
""")

# Configurações do ambiente
states = ["Alta Demanda", "Média Demanda", "Baixa Demanda"]
actions = ["Preço Baixo", "Preço Médio", "Preço Alto"]
n_states = len(states)
n_actions = len(actions)

# Inicializar Q-Table
Q = np.zeros((n_states, n_actions))

# Hiperparâmetros ajustáveis pelo aluno
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

# Treinamento
for _ in range(episodios):
    state = np.random.randint(0, n_states)
    if np.random.rand() < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(Q[state])

    reward = obter_recompensa(state, action)
    next_state = np.random.randint(0, n_states)

    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

# Mostrar Q-Table e política aprendida
df_q = pd.DataFrame(Q, index=states, columns=actions)
st.subheader("Q-Table final aprendida")
st.dataframe(df_q.style.format("{:.2f}"))

# Melhor ação por estado
policy = df_q.idxmax(axis=1).rename("Melhor Preço Sugerido")
st.subheader("Política aprendida (Melhor preço por nível de demanda)")
st.dataframe(policy)

st.markdown("""
### 📌 Como usar a Q-Table no e-commerce?
- A Q-Table mostra o valor esperado de lucro para cada ação em cada situação de demanda.
- O e-commerce poderia usar a **melhor ação sugerida** (preço) como orientação para seu sistema de precificação automática.
- Exemplo: Se a demanda está alta → o sistema aplica o preço alto, porque foi o que o agente aprendeu que maximiza o lucro.

### 🎯 Atividade para o aluno
✅ Execute o treinamento e observe a política aprendida.  
✅ Explique:
- Qual preço o agente recomenda para cada nível de demanda?
- Por que essa política faz sentido (ou não) para o e-commerce?
- Como você ajustaria os parâmetros para obter uma política diferente?

💡 **Dica:** Teste diferentes valores de ε, α e γ para entender o impacto no aprendizado!
""")
