
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import Counter

st.title("Sistema de Recomendação por Perfil de Afinidade (KNN)")

st.markdown("Faça o upload das duas bases: a base de histórico de clientes (com categoria comprada) e a base de novos clientes a serem classificados.")

# Upload das bases
historico_file = st.file_uploader("1. Base de Treinamento (com categoria_comprada)", type=["csv"])
novos_file = st.file_uploader("2. Base de Novos Clientes (sem categoria)", type=["csv"])

# Seleção do valor de K
k_valor = st.slider("Escolha o valor de K para o modelo KNN:", min_value=3, max_value=6, value=5)

if historico_file and novos_file:
    historico = pd.read_csv(historico_file)
    novos = pd.read_csv(novos_file)

    st.subheader("Pré-visualização da base de treinamento")
    st.dataframe(historico.head())

    st.subheader("Pré-visualização da base de novos clientes")
    st.dataframe(novos.head())

    # Features utilizadas
    features = [
        'idade', 'num_compras', 'valor_medio_compra', 'avaliacao_media',
        'Tecnologia', 'Moda', 'Livros', 'Esporte', 'Beleza', 'Casa e Jardim', 'Brinquedos'
    ]

    X_train = historico[features]
    y_train = historico['categoria_comprada']
    X_test = novos[features]

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo KNN
    modelo = KNeighborsClassifier(n_neighbors=k_valor)
    modelo.fit(X_train_scaled, y_train)

    # Previsão
    previsoes = modelo.predict(X_test_scaled)
    probabilidades = modelo.predict_proba(X_test_scaled)
    novos['categoria_recomendada'] = previsoes
    novos['grau_confianca'] = np.max(probabilidades, axis=1)

    # Cliente individual
    cliente_escolhido = st.selectbox("Selecione um cliente para ver detalhes:", novos['cliente_id'])
    idx = novos.index[novos['cliente_id'] == cliente_escolhido].tolist()[0]

    st.subheader("Resultado do cliente selecionado")
    st.write(novos.loc[[idx], ['cliente_id', 'categoria_recomendada', 'grau_confianca']])

    # Votos dos vizinhos
    distancias, indices = modelo.kneighbors([X_test_scaled[idx]])
    votos = y_train.iloc[indices[0]].tolist()
    contagem_votos = Counter(votos)

    st.markdown("**Votos dos vizinhos mais próximos:**")
    for categoria, count in contagem_votos.items():
        st.write(f"{categoria}: {count} voto(s)")

    # Resultado completo
    st.subheader("Resultado completo da recomendação")
    st.dataframe(novos[['cliente_id', 'categoria_recomendada', 'grau_confianca']])

    # 🔍 Resumo por categoria
    st.subheader("Resumo por categoria recomendada")
    resumo = novos['categoria_recomendada'].value_counts().reset_index()
    resumo.columns = ['Categoria', 'Total de Clientes']
    st.dataframe(resumo)

    # Download
    csv_resultado = novos.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV com previsões", data=csv_resultado, file_name="recomendacoes.csv", mime="text/csv")
