
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import numpy as np

st.title("Sistema 2 – Predição da Peça com Problema (Rede Neural MLP - scikit-learn)")

st.markdown("Este sistema utiliza uma rede neural com camadas ocultas (MLP) para prever qual peça está mais propensa a falhar em máquinas que exigem manutenção preventiva.")

# Upload das bases
base_treino = st.file_uploader("1. Upload da base de treinamento (com 'peca_com_problema')", type=["csv"])
base_alvo = st.file_uploader("2. Upload da base com máquinas que precisam de manutenção", type=["csv"])

if base_treino and base_alvo:
    df_treino = pd.read_csv(base_treino)
    df_alvo = pd.read_csv(base_alvo)

    st.subheader("Amostra da base de treinamento")
    st.dataframe(df_treino.head())

    st.subheader("Amostra da base das máquinas a analisar")
    st.dataframe(df_alvo.head())

    # Features
    features = ['temperatura_media', 'nivel_de_vibracao', 'pressao_do_sistema',
                'horas_de_uso', 'idade_da_maquina'] +                [f"tipo_peca_{i+1}" for i in range(5)] +                [f"historico_trocas_peca_{i+1}" for i in range(5)]

    # Unir dados para dummificação
    df_all = pd.concat([df_treino[features], df_alvo[features]], axis=0)
    df_all = pd.get_dummies(df_all, columns=[f"tipo_peca_{i+1}" for i in range(5)])

    X_train = df_all.iloc[:len(df_treino), :]
    X_pred = df_all.iloc[len(df_treino):, :]

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # Codificação do rótulo
    le = LabelEncoder()
    y_train = le.fit_transform(df_treino['peca_com_problema'])

    # Modelo MLP (rede neural rasa)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Previsões
    pred_probs = model.predict_proba(X_pred_scaled)
    pred_labels = model.predict(X_pred_scaled)
    categorias_previstas = le.inverse_transform(pred_labels)

    df_alvo['peca_predita'] = categorias_previstas
    df_alvo['confianca'] = np.max(pred_probs, axis=1)

    st.subheader("Resultado da predição de peças com problema")
    st.dataframe(df_alvo[['id_maquina', 'peca_predita', 'confianca']])

    # Download
    csv_resultado = df_alvo.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV com peças previstas", data=csv_resultado, file_name="pecas_com_problema.csv", mime="text/csv")
