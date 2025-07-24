
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

st.title("Sistema 2 – Predição da Peça com Problema (Rede Neural)")

st.markdown("Este sistema utiliza uma rede neural com camadas ocultas para prever qual peça está mais propensa a falhar em máquinas que já exigem manutenção preventiva.")

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

    # Features e pré-processamento
    features = ['temperatura_media', 'nivel_de_vibracao', 'pressao_do_sistema',
                'horas_de_uso', 'idade_da_maquina'] +                [f"tipo_peca_{i+1}" for i in range(5)] +                [f"historico_trocas_peca_{i+1}" for i in range(5)]

    # Unir dados para dummyficar as categorias
    df_all = pd.concat([df_treino[features], df_alvo[features]], axis=0)
    df_all = pd.get_dummies(df_all, columns=[f"tipo_peca_{i+1}" for i in range(5)])

    X_train = df_all.iloc[:len(df_treino), :]
    X_pred = df_all.iloc[len(df_treino):, :]

    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # Encoder para rótulo
    le = LabelEncoder()
    y_train = le.fit_transform(df_treino['peca_com_problema'])

    # Construção da rede neural
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treinamento
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)

    # Predição
    pred_probs = model.predict(X_pred_scaled)
    pred_labels = np.argmax(pred_probs, axis=1)
    categorias_previstas = le.inverse_transform(pred_labels)

    df_alvo['peca_predita'] = categorias_previstas
    df_alvo['confianca'] = np.max(pred_probs, axis=1)

    st.subheader("Resultado da predição de peças com problema")
    st.dataframe(df_alvo[['id_maquina', 'peca_predita', 'confianca']])

    # Download
    csv_resultado = df_alvo.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV com peças previstas", data=csv_resultado, file_name="pecas_com_problema.csv", mime="text/csv")
