
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title("Sistema 1 – Previsão de Manutenção Preventiva")

st.markdown("Este sistema classifica quais máquinas precisam de manutenção preventiva com base em sensores e composição de peças.")

# Upload das bases
base_treino = st.file_uploader("1. Upload da base de treinamento (com 'falha_ocorreu')", type=["csv"])
base_atuais = st.file_uploader("2. Upload da base de máquinas atuais (sem rótulo)", type=["csv"])

if base_treino and base_atuais:
    df_treino = pd.read_csv(base_treino)
    df_atuais = pd.read_csv(base_atuais)

    st.subheader("Amostra da base de treinamento")
    st.dataframe(df_treino.head())

    st.subheader("Amostra da base de máquinas atuais")
    st.dataframe(df_atuais.head())

    features = ['temperatura_media', 'nivel_de_vibracao', 'pressao_do_sistema',
                'horas_de_uso', 'idade_da_maquina'] +                [f"tipo_peca_{i+1}" for i in range(5)] +                [f"historico_trocas_peca_{i+1}" for i in range(5)]

    # Converter colunas categóricas
    df_all = pd.concat([df_treino[features], df_atuais[features]], axis=0)
    df_all = pd.get_dummies(df_all, columns=[f"tipo_peca_{i+1}" for i in range(5)])

    X_train = df_all.iloc[:len(df_treino), :]
    X_test = df_all.iloc[len(df_treino):, :]
    y_train = df_treino['falha_ocorreu']

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train_scaled, y_train)

    # Previsão
    previsoes = modelo.predict(X_test_scaled)
    probas = modelo.predict_proba(X_test_scaled)[:, 1]
    df_atuais['falha_prevista'] = previsoes
    df_atuais['grau_confianca'] = probas

    st.subheader("Resultado completo da previsão")
    st.dataframe(df_atuais[['id_maquina', 'falha_prevista', 'grau_confianca']])

    # Filtrar apenas as que precisam de manutenção
    df_manutencao = df_atuais[df_atuais['falha_prevista'] == 1]
    st.subheader("Máquinas com manutenção preventiva recomendada")
    st.dataframe(df_manutencao)

    # Download
    csv = df_manutencao.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar máquinas que precisam de manutenção", data=csv, file_name="maquinas_para_manutencao.csv", mime="text/csv")
