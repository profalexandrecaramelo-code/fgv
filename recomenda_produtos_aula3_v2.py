import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import Counter

st.title("Recomendação de Categorias de Compra com Algoritmo Supervisionado")
st.markdown("Utilizar um sistema de IA baseado em KNN para recomendar categorias de compra a partir do perfil de clientes atuais, explorando o impacto do valor de K, o grau de confiança da previsão e a lógica de decisão dos vizinhos mais próximos.")
st.markdown ("Você está atuando na equipe de marketing de uma empresa de e-commerce e precisa identificar a categoria de produto mais adequada para um grupo de novos clientes que ainda não realizaram compras. Para isso, um modelo de IA será treinado com o histórico de compras de 900 clientes reais e agora deve ser usado para prever o comportamento dos clientes novos.")

st.markdown("Faça o upload de duas bases de dados:")

# Upload das bases
historico_file = st.file_uploader("1. Base de Histórico de Compras (com rótulo de categoria)", type=["csv"])
atuais_file = st.file_uploader("2. Base de Clientes Atuais (sem rótulo)", type=["csv"])

# Seleção do valor de K
k_valor = st.slider("Selecione o valor de K para o modelo KNN:", min_value=3, max_value=6, value=5)

if historico_file and atuais_file:
    historico = pd.read_csv(historico_file)
    atuais = pd.read_csv(atuais_file)

    st.subheader("Pré-visualização da base de histórico")
    st.dataframe(historico.head())

    st.subheader("Pré-visualização da base de clientes atuais")
    st.dataframe(atuais.head())

    # Definir colunas numéricas para treino
    features = ['idade', 'num_compras', 'valor_medio_compra', 
                'avaliacao_media', 'tempo_medio_sessao_min', 'num_itens_visualizados']

    X_train = historico[features]
    y_train = historico['categoria_comprada']
    X_test = atuais[features]

    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar o modelo KNN com K definido pelo aluno
    modelo = KNeighborsClassifier(n_neighbors=k_valor)
    modelo.fit(X_train_scaled, y_train)

    # Fazer previsões e obter probabilidades
    previsoes = modelo.predict(X_test_scaled)
    probabilidades = modelo.predict_proba(X_test_scaled)
    atuais['categoria_recomendada'] = previsoes
    atuais['grau_confianca'] = np.max(probabilidades, axis=1)

    # Selecionar um cliente para exibição individual
    cliente_escolhido = st.selectbox("Selecione um cliente atual para visualizar a recomendação:", atuais['cliente_id'])
    cliente_detalhe = atuais[atuais['cliente_id'] == cliente_escolhido]

    st.subheader("Recomendação para o cliente selecionado")
    st.write(cliente_detalhe[['cliente_id', 'categoria_recomendada', 'grau_confianca']])

    # Mostrar os votos dos vizinhos
    cliente_idx = atuais.index[atuais['cliente_id'] == cliente_escolhido].tolist()[0]
    distancias, indices = modelo.kneighbors([X_test_scaled[cliente_idx]])
    votos = y_train.iloc[indices[0]].tolist()
    contagem_votos = Counter(votos)

    st.markdown("**Distribuição dos votos dos vizinhos:**")
    for categoria, count in contagem_votos.items():
        st.write(f"{categoria}: {count} voto(s)")

    # Download da base com recomendações
    csv_resultado = atuais.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar resultados como CSV", data=csv_resultado, file_name="clientes_recomendados.csv", mime="text/csv")
