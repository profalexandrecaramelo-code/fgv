
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Criando dataset
data = {
    'Idade': [25, 28, 45, 35, 50, 23, 40, 60, 48, 33, 55, 62, 21, 31, 29, 41, 47, 53,
              27, 30, 34, 38, 44, 49, 26, 32, 43, 57, 61, 36, 39, 46, 52, 59, 63, 37],
    'Renda': [50000, 53000, 60000, 80000, 120000, 30000, 70000, 150000, 110000, 65000, 140000, 123000,
              28000, 72000, 49000, 85000, 95000, 133000, 40000, 67000, 73000, 77000, 89000, 125000,
              32000, 58000, 91000, 118000, 144000, 82000, 75000, 96000, 137000, 148000, 130000, 87000],
    'Endividamento': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0,
                      1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
                      1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    'Dependentes': [0, 1, 2, 1, 3, 0, 1, 5, 1, 1, 3, 4,
                    2, 0, 1, 2, 3, 4, 0, 1, 1, 2, 2, 3,
                    1, 1, 2, 3, 5, 1, 3, 4, 2, 3, 6, 2],
    'Comprou': [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
                1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1,
                0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

# Separando variáveis
X = df[['Idade', 'Renda', 'Endividamento', 'Dependentes']]
y = df['Comprou']

# Divisão de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Avaliação
acc = accuracy_score(y_test, tree.predict(X_test))
cv_score = cross_val_score(tree, X, y, cv=5).mean()

# Interface
st.title("🌳 Previsão de Compra com Árvore de Decisão")
st.write(f"Acurácia no teste: **{acc:.2f}**")
st.write(f"Acurácia média com validação cruzada: **{cv_score:.2f}**")

st.subheader("🧾 Informe os dados do cliente:")

idade = st.number_input("Idade", min_value=18, max_value=100, value=30)
renda = st.number_input("Renda", min_value=10000, max_value=200000, value=50000)
endividamento = st.selectbox("Endividamento", options=[0, 1], format_func=lambda x: "Baixo" if x == 0 else "Alto")
dependentes = st.number_input("Número de dependentes", min_value=0, max_value=10, value=1)

if st.button("🔍 Prever"):
    nova_instancia = pd.DataFrame({
        'Idade': [idade],
        'Renda': [renda],
        'Endividamento': [endividamento],
        'Dependentes': [dependentes]
    })

    previsao = tree.predict(nova_instancia)
    if previsao[0] == 1:
        st.success("✅ O modelo prevê que o cliente **compraria** o produto.")
    else:
        st.error("❌ O modelo prevê que o cliente **não compraria** o produto.")
