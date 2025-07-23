
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import Counter

st.title("Previsão de Falhas em Equipamentos Industriais")

st.markdown("Faça o upload das duas bases: histórico de falhas (com rótulo) e situação atual das máquinas (sem rótulo).")

# Upload das bases
historico_file = st.file_uploader("1. Base de Histórico de Falhas (com rótulo falha_ocorreu)", type=["csv"])
atuais_file = st.file_uploader("2. Base de Máquinas Atuais (sem rótulo)", type=["csv"])

# Seleção do valor de K
k_valor = st.slider("Escolha o valor de K para o modelo KNN:", min_value=3, max_value=6, value=5)

if historico_file and atuais_file:
    historico = pd.read_csv(historico_file)
    atuais = pd.read_csv(atuais_file)

    st.subheader("Pré-visualização do histórico de falhas")
    st.dataframe(historico.head())

    st.subheader("Pré-visualização das máquinas atuais")
    st.dataframe(atuais.head())

    features = ['temperatura_media', 'nivel_de_vibracao', 'pressao_do_sistema', 'horas_de_uso', 'idade_da_maquina']
    X_train = historico[features]
    y_train = historico['falha_ocorreu']
    X_test = atuais[features]

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo KNN
    modelo = KNeighborsClassifier(n_neighbors=k_valor)
    modelo.fit(X_train_scaled, y_train)

    # Previsões e probabilidade
    previsoes = modelo.predict(X_test_scaled)
    probabilidades = modelo.predict_proba(X_test_scaled)
    atuais['falha_prevista'] = previsoes
    atuais['grau_confianca'] = np.max(probabilidades, axis=1)

    # Seleção de máquina
    maquina_escolhida = st.selectbox("Selecione uma máquina para ver detalhes:", atuais['id_maquina'])
    idx = atuais.index[atuais['id_maquina'] == maquina_escolhida].tolist()[0]

    st.subheader("Resultado da máquina selecionada")
    st.write(atuais.loc[[idx], ['id_maquina', 'falha_prevista', 'grau_confianca']])

    # Mostrar votos dos vizinhos
    distancias, indices = modelo.kneighbors([X_test_scaled[idx]])
    votos = y_train.iloc[indices[0]].tolist()
    contagem_votos = Counter(votos)

    st.markdown("**Votos dos vizinhos mais próximos:**")
    for classe, contagem in contagem_votos.items():
        st.write(f"Classe {classe} (falha: {'sim' if classe == 1 else 'não'}): {contagem} voto(s)")

    # Tabela geral
    st.subheader("Resultado completo das 40 máquinas")
    st.dataframe(atuais[['id_maquina', 'falha_prevista', 'grau_confianca']])

    # Download
    csv_resultado = atuais.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV com previsões", data=csv_resultado, file_name="resultado_falhas.csv", mime="text/csv")
