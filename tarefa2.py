
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(page_title="Campanha de E-commerce — Avaliação (foco em Recall)", layout="wide")

st.title("🛍️ Campanha de E-commerce — Avaliação de Classificação (foco em Recall)")

st.markdown(
    """
    **Cenário:** e-commerce de moda com campanha por e-mail. Queremos **não perder compradores reais**, então priorizamos **Recall** (sensibilidade).  
    Envie a **base de treino/validação** (com a coluna alvo `comprou_pos_campanha`) e a **base de predição** (sem alvo).
    """
)

with st.expander("ℹ️ O que este app faz?", expanded=True):
    st.markdown(
        """
        1. **Treina e valida** um classificador (Logistic Regression).  
        2. Calcula **Acurácia, Precisão e Recall** na **validação**.  
        3. **Não** exibe o cálculo detalhado; os alunos devem **reproduzir as contas**.  
        4. Gera **predições** para a base **sem rótulo** e **mostra apenas quem o modelo prevê que comprará**, com destaque do **canal preferido**.
        """
    )

with st.sidebar:
    st.header("Parâmetros")
    test_size = st.slider("Proporção de teste (validação)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    standardize_numeric = st.checkbox("Padronizar variáveis numéricas (StandardScaler)", value=True)

st.subheader("1) Base de **Treino/Validação**")
train_file = st.file_uploader("Envie CSV com a coluna alvo `comprou_pos_campanha`", type=["csv"], key="train")
target_col = "comprou_pos_campanha"
df_train = None
if train_file is not None:
    try:
        df_train = pd.read_csv(train_file)
    except Exception:
        train_file.seek(0)
        df_train = pd.read_csv(train_file, sep=";")
    st.write("Pré-visualização (treino/validação):", df_train.head())
    if target_col not in df_train.columns:
        st.error(f"A base deve conter a coluna alvo `{target_col}`.")
        df_train = None

st.subheader("2) Base de **Predição** (sem rótulo)")
pred_file = st.file_uploader("Envie CSV **sem** a coluna alvo", type=["csv"], key="pred")
df_pred = None
if pred_file is not None:
    try:
        df_pred = pd.read_csv(pred_file)
    except Exception:
        pred_file.seek(0)
        df_pred = pd.read_csv(pred_file, sep=";")
    st.write("Pré-visualização (predição):", df_pred.head())

if df_train is not None:
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler() if standardize_numeric else "passthrough", num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Único algoritmo: Logistic Regression
    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Métricas (sem detalhamento)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia", f"{acc:.3f}")
    c2.metric("Precisão", f"{prec:.3f}")
    c3.metric("Recall (foco)", f"{rec:.3f}")

    st.markdown("---")
    st.markdown("### 🔮 Predições (apenas **comprará**)")
    if df_pred is not None:
        X_deploy = df_pred.copy()
        y_pred_deploy = pipe.predict(X_deploy)

        df_out = df_pred.copy()
        df_out["Predicao"] = y_pred_deploy

        # Filtrar apenas os previstos como 1 (comprará)
        df_pos = df_out[df_out["Predicao"] == 1].copy()

        if df_pos.empty:
            st.info("Nenhum cliente previsto como 'comprará' para a base enviada.")
        else:
            # Destaque azul na coluna 'canal_preferido' usando pandas Styler
            def highlight_channel_col(s):
                # color entire column if it's 'canal_preferido'
                return ['background-color: #D0E8FF' if s.name == 'canal_preferido' else '' for _ in s]

            styled = df_pos.style.apply(highlight_channel_col, axis=0)
            st.dataframe(styled, use_container_width=True)

            # Download apenas do grupo previsto como comprará
            csv_bytes = df_pos.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar predições (apenas 'comprará')", data=csv_bytes, file_name="predicoes_comprara.csv", mime="text/csv")

else:
    st.info("Envie a base de treino/validação (com `comprou_pos_campanha`) para continuar.")

st.markdown("---")
st.caption("Lembrete: **Recall = VP / (VP + FN)**. Os alunos devem reproduzir os cálculos detalhados com base nas saídas do app.")
