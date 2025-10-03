
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import numpy as np

st.title("Sistem2 – Defective part prediction (Neural Network MLP - scikit-learn)")

st.markdown("This system uses a Multi-layer perceptron (MLP) to predict which part is most likely to fail in machines requiring preventive maintenance.")

# Dataset Upload
base_treino = st.file_uploader("1. Upload training dataset (with 'defective_part')", type=["csv"])
base_alvo = st.file_uploader("2. Upload dataset with machines needing maintenance", type=["csv"])

if base_treino and base_alvo:
    df_treino = pd.read_csv(base_treino)
    df_alvo = pd.read_csv(base_alvo)

    st.subheader("Sample of trainning dataset")
    st.dataframe(df_treino.head())

    st.subheader("Sample of dataset with machines needing maintenance")
    st.dataframe(df_alvo.head())

    # Features
    features = ['average_temperature', 'vibration_level', 'system_pressure',
                'usage_hours', 'machine_age'] +                [f"part_type_{i+1}" for i in range(5)] +                [f"part_replacement_history_{i+1}" for i in range(5)]

    # Merge data for dummyfication
    df_all = pd.concat([df_treino[features], df_alvo[features]], axis=0)
    df_all = pd.get_dummies(df_all, columns=[f"tipo_peca_{i+1}" for i in range(5)])

    X_train = df_all.iloc[:len(df_treino), :]
    X_pred = df_all.iloc[len(df_treino):, :]

    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(df_treino['defective_part'])

    # Modelo MLP (rede neural rasa)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Previsões
    pred_probs = model.predict_proba(X_pred_scaled)
    pred_labels = model.predict(X_pred_scaled)
    categorias_previstas = le.inverse_transform(pred_labels)

    df_alvo['part_predicted'] = categorias_previstas
    df_alvo['trust'] = np.max(pred_probs, axis=1)

    st.subheader("Results - Defective part prediction")
    st.dataframe(df_alvo[['id_machine', 'part_predicted', 'trust']])

    # Download
    csv_resultado = df_alvo.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV with predicted parts", data=csv_resultado, file_name="predicted_parts.csv", mime="text/csv")
