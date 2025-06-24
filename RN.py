
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Título do app
st.title("🌟 Rede Neural Básica (MLPClassifier) - IA aplicada aos Negócios")

st.write("""
Este app demonstra o funcionamento de uma Rede Neural Artificial simples (MLPClassifier) 
para resolver um problema de classificação. 
Experimente alterar os parâmetros e observe o impacto na acurácia e na fronteira de decisão!
""")

# Parâmetros configuráveis
n_neurons_layer1 = st.slider('Número de neurônios na 1ª camada oculta', 1, 20, 5)
n_neurons_layer2 = st.slider('Número de neurônios na 2ª camada oculta (0 para desativar)', 0, 20, 2)
alpha = st.slider('Alpha (regularização)', 0.0001, 0.1, 0.01, step=0.0001, format="%.4f")
max_iter = st.slider('Número máximo de iterações', 100, 5000, 1000, step=100)

# Gerando dados
X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Arquitetura da rede
if n_neurons_layer2 == 0:
    hidden_layers = (n_neurons_layer1,)
else:
    hidden_layers = (n_neurons_layer1, n_neurons_layer2)

# Modelo
clf = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=max_iter, alpha=alpha, random_state=42)
clf.fit(X_train, y_train)

# Previsão e acurácia
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.markdown(f"### ✅ Acurácia no conjunto de teste: **{acc:.2f}**")

# Fronteira de decisão
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 500),
                     np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 500))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu)
ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.RdBu, edgecolor='k', marker='o', label='Treino')
ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=plt.cm.RdBu, edgecolor='k', marker='^', label='Teste')
ax.legend()
ax.set_title("Fronteira de decisão aprendida pela Rede Neural")

st.pyplot(fig)
