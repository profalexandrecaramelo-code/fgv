
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Detecção de Fraude — Precisão vs Recall (Exercício)", layout="wide")

st.title("🛡️ Detecção de Fraude — Exercício Prático (Precisão × Recall)")

st.markdown(
    """
    **Objetivo do exercício:** entender o **trade-off entre Precisão e Recall** ajustando o **limiar de decisão** do mesmo modelo.
    - **Config A (foco em Recall):** reduzir fraudes que passam (FN).
    - **Config B (foco em Precisão):** reduzir alarmes falsos (FP).

    Envie uma **base de treino/validação** (com a coluna alvo `fraude`) e, opcionalmente, uma **base de predição** (sem alvo).
    Se preferir, clique em **Usar dados de exemplo** para carregar dados sintéticos.
    """
)

with st.sidebar:
    st.header("Parâmetros do exercício")
    test_size = st.slider("Proporção de teste (validação)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    standardize_numeric = st.checkbox("Padronizar variáveis numéricas (StandardScaler)", value=True)
    st.markdown("---")
    st.subheader("Limiar de decisão (probabilidade ≥ limiar ⇒ FRAUDE)")
    th_recall = st.slider("Config A — foco em Recall", 0.0, 1.0, 0.30, 0.01)
    th_precision = st.slider("Config B — foco em Precisão", 0.0, 1.0, 0.70, 0.01)
    st.markdown("---")
    st.subheader("Custos (aprox.)")
    loss_per_fraud = st.number_input("Perda média se a fraude passar (R$)", min_value=0.0, value=600.0, step=50.0)
    review_cost = st.number_input("Custo por análise manual (R$)", min_value=0.0, value=5.0, step=1.0)

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)        # sklearn <1.2

# Uploads
st.subheader("1) Base de **Treino/Validação**")
c1, c2 = st.columns([2,1])
with c1:
    train_file = st.file_uploader("Envie CSV com a coluna alvo `fraude` (0/1)", type=["csv"], key="train")
with c2:
    use_example = st.button("Usar dados de exemplo")

df_train = None
target_col = "fraude"

if use_example and train_file is None:
    st.info("Carregando dados de exemplo embutidos…")
    try:
        df_train = pd.read_csv("fraude_treino_1000.csv")
    except Exception:
        st.warning("Dados de exemplo não encontrados no repositório. Faça upload manual do CSV com a coluna `fraude`.")
elif train_file is not None:
    try:
        df_train = pd.read_csv(train_file)
    except Exception:
        train_file.seek(0)
        df_train = pd.read_csv(train_file, sep=";")

if df_train is not None:
    st.write("Pré-visualização (treino/validação):", df_train.head())
    if target_col not in df_train.columns:
        st.error(f"A base deve conter a coluna alvo `{target_col}`.")
        df_train = None

st.subheader("2) Base de **Predição** (opcional, sem alvo)")
pred_file = st.file_uploader("Envie CSV **sem** a coluna `fraude`", type=["csv"], key="pred")
df_pred = None
if pred_file is not None:
    try:
        df_pred = pd.read_csv(pred_file)
    except Exception:
        pred_file.seek(0)
        df_pred = pd.read_csv(pred_file, sep=";")
    st.write("Pré-visualização (predição):", df_pred.head())

# Train & Evaluate
if df_train is not None:
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []
    if cat_cols:
        transformers.append(("cat", make_ohe(), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler() if standardize_numeric else "passthrough", num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    def eval_with_threshold(th):
        y_pred = (y_proba >= th).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        vn, fp, fn, vp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
        total_cost = fn * loss_per_fraud + (vp + fp) * review_cost
        return acc, prec, rec, vn, fp, fn, vp, total_cost, cm.sum()

    accA, precA, recA, vnA, fpA, fnA, vpA, costA, totA = eval_with_threshold(th_recall)
    accB, precB, recB, vnB, fpB, fnB, vpB, costB, totB = eval_with_threshold(th_precision)

    st.markdown("### ✅ Resultados na validação (mesmo modelo, limiares diferentes)")
    colA, colB = st.columns(2, gap="large")
    with colA:
        st.subheader("Config A — foco em Recall")
        st.metric("Recall", f"{recA:.3f}")
        st.metric("Precisão", f"{precA:.3f}")
        st.metric("Acurácia", f"{accA:.3f}")
        st.markdown("**Matriz de confusão (valores)**")
        df_cm_A = pd.DataFrame([[vnA, fpA],[fnA, vpA]], index=["Real 0","Real 1"], columns=["Prev 0","Prev 1"])
        st.table(df_cm_A)
        st.caption(f"VP={vpA} • FP={fpA} • FN={fnA} • VN={vnA} • Total={totA}")
        st.markdown(f"**Custo estimado do dia:** R$ {costA:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))

    with colB:
        st.subheader("Config B — foco em Precisão")
        st.metric("Recall", f"{recB:.3f}")
        st.metric("Precisão", f"{precB:.3f}")
        st.metric("Acurácia", f"{accB:.3f}")
        st.markdown("**Matriz de confusão (valores)**")
        df_cm_B = pd.DataFrame([[vnB, fpB],[fnB, vpB]], index=["Real 0","Real 1"], columns=["Prev 0","Prev 1"])
        st.table(df_cm_B)
        st.caption(f"VP={vpB} • FP={fpB} • FN={fnB} • VN={vnB} • Total={totB}")
        st.markdown(f"**Custo estimado do dia:** R$ {costB:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))

    st.markdown("---")
    st.markdown("### 🔮 Predições na base **sem rótulo** (opcional)")
    if df_pred is not None:
        proba_pred = pipe.predict_proba(df_pred)[:, 1]
        df_pred_out = df_pred.copy()
        df_pred_out["proba_fraude"] = proba_pred

        tabA, tabB = st.tabs(["Config A (foco em Recall)", "Config B (foco em Precisão)"])
        with tabA:
            dfA = df_pred_out[df_pred_out["proba_fraude"] >= th_recall].copy()
            dfA["flag_fraude_A"] = 1
            st.write("Previstos como FRAUDE (Config A):", dfA.shape[0])
            st.dataframe(dfA.head(100), use_container_width=True)
            st.download_button("⬇️ Baixar FRAUDES — Config A", data=dfA.to_csv(index=False).encode("utf-8"),
                               file_name="fraudes_configA.csv", mime="text/csv")
        with tabB:
            dfB = df_pred_out[df_pred_out["proba_fraude"] >= th_precision].copy()
            dfB["flag_fraude_B"] = 1
            st.write("Previstos como FRAUDE (Config B):", dfB.shape[0])
            st.dataframe(dfB.head(100), use_container_width=True)
            st.download_button("⬇️ Baixar FRAUDES — Config B", data=dfB.to_csv(index=False).encode("utf-8"),
                               file_name="fraudes_configB.csv", mime="text/csv")

else:
    st.info("Envie a base de treino/validação (com `fraude`) ou clique em **Usar dados de exemplo** para começar.")

st.markdown("---")
with st.expander("📌 Guia do exercício (copie para o relatório)", expanded=True):
    st.markdown(
        """
        **Passos:**
        1) Carregue a base de treino (ou use os dados de exemplo) e defina os **limiares** das Configurações A e B.  
        2) Compare **Precisão, Recall, Acurácia** e a **matriz de confusão** das duas configurações.  
        3) Preencha o quadro de **custos** com seus valores (perda por fraude que passa; custo de revisão) e compare o **custo estimado**.  
        4) (Opcional) Carregue a base de **predição**, baixe a lista de casos **marcados como fraude** em cada configuração.

        **Perguntas para responder no relatório:**
        - a) Qual configuração reduz mais a **perda total**? Justifique com os números.  
        - b) Mostre o **trade-off**: o que acontece com FP, FN, Precisão e Recall quando você muda o **limiar**?  
        - c) Proponha uma **política operacional** (ex.: usar Config A no pico, B fora do pico; ou thresholds por valor de compra).  
        - d) Que **novas variáveis** ajudariam a melhorar Recall **sem** explodir FP? Dê 2 exemplos.
        """
    )
