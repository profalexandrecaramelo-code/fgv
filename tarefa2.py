
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Campanha de E-commerce — Métricas (Acurácia, Precisão, Recall)", layout="wide")

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
        3. **Explica como cada métrica foi obtida**, exibindo os valores usados no cálculo.  
        4. Gera **predições** para a base **sem rótulo** e disponibiliza para download.
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
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
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

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    # Consideramos 1 como a classe "comprou"
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Valores para a explicação (sem exibir a matriz gráfica)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # Mapear VP, FP, FN, VN para binário {0,1} se possível
    # Assumindo labels ordenados; se binário (0,1): 
    # linha = real, coluna = previsto
    vp = fp = fn = vn = None
    if set(labels) == {0,1}:
        vn = int(cm[0,0])
        fp = int(cm[0,1])
        fn = int(cm[1,0])
        vp = int(cm[1,1])

    st.markdown("### ✅ Métricas na validação")
    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia", f"{acc:.3f}")
    c2.metric("Precisão", f"{prec:.3f}")
    c3.metric("Recall (foco)", f"{rec:.3f}")

    st.markdown("### 📘 Como o sistema chegou a esses valores")
    if vp is not None:
        total = vn + fp + fn + vp
        st.markdown(
            f"""
            **Definições (classe positiva = 1 — *comprou_pos_campanha*):**  
            • **VP (Verdadeiro Positivo)**: previu **1** e o real era **1** → **{vp}**  
            • **FP (Falso Positivo)**: previu **1** e o real era **0** → **{fp}**  
            • **FN (Falso Negativo)**: previu **0** e o real era **1** → **{fn}**  
            • **VN (Verdadeiro Negativo)**: previu **0** e o real era **0** → **{vn}**  
            • **Total**: **{total}**

            **Fórmulas aplicadas com os valores acima:**  
            • **Acurácia** = (VP + VN) / Total = ({vp} + {vn}) / {total} = **{(vp+vn)/total:.3f}**  
            • **Precisão** = VP / (VP + FP) = {vp} / ({vp} + {fp}) = **{(vp/(vp+fp) if (vp+fp)>0 else 0):.3f}**  
            • **Recall** = VP / (VP + FN) = {vp} / ({vp} + {fn}) = **{(vp/(vp+fn) if (vp+fn)>0 else 0):.3f}**
            """
        )
    else:
        st.info("Métricas explicadas: para problemas multiclasse, as fórmulas são generalizadas (média macro).")

    st.markdown("---")
    st.markdown("### 🔮 Predições para a base **sem rótulo**")
    if df_pred is not None:
        X_deploy = df_pred.copy()
        y_pred_deploy = pipe.predict(X_deploy)

        # Saída final exibida no sistema
        df_out = df_pred.copy()
        df_out["Predicao"] = y_pred_deploy
        st.dataframe(df_out.head(50))
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar predições (.csv)", data=csv_bytes, file_name="predicoes.csv", mime="text/csv")

else:
    st.info("Envie a base de treino/validação (com `comprou_pos_campanha`) para continuar.")

st.markdown("---")
st.caption("Lembrete: **Recall = VP / (VP + FN)**. Em campanhas, perder compradores reais (FN) é mais caro; por isso priorizamos sensibilidade.")
