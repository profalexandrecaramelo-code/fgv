
import io
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="Avaliação de Modelo (Precisão/Recall/Acurácia)", layout="wide")

st.title("🏷️ Classificação supervisionada com duas bases (treino/validação e predição)")
st.write(
    """
    Este app permite **treinar, validar e avaliar** um classificador supervisionado com uma base de **treino/validação**
    (onde são calculadas Acurácia, Precisão e Recall e exibida a **matriz de confusão**)
    e, em seguida, fazer **predições** em uma segunda base **sem rótulo** (base de predição).
    """
)

with st.sidebar:
    st.header("Configurações")
    model_name = st.selectbox(
        "Algoritmo",
        ["Logistic Regression", "Random Forest", "KNN"],
        help="Escolha o algoritmo de classificação."
    )
    test_size = st.slider("Proporção de teste (validação)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random seed (reprodutibilidade)", min_value=0, value=42, step=1)
    standardize_numeric = st.checkbox("Padronizar variáveis numéricas (StandardScaler)", value=True)
    n_neighbors = st.slider("K (apenas para KNN)", 1, 25, 5, 1)
    rf_n_estimators = st.slider("Árvores (apenas para Random Forest)", 50, 500, 200, 50)
    rf_max_depth = st.select_slider("Profundidade máxima (RF)", options=[None, 5, 10, 20, 30], value=None)

st.subheader("1) Base de **Treino/Validação**")
train_file = st.file_uploader("Envie um CSV para treino/validação (com a coluna alvo)", type=["csv"], key="train")

target_col = None
df_train = None
if train_file is not None:
    try:
        df_train = pd.read_csv(train_file)
    except Exception:
        train_file.seek(0)
        df_train = pd.read_csv(train_file, sep=";")
    st.write("Pré-visualização da base de treino/validação:", df_train.head())

    # Escolha da coluna alvo
    cols = list(df_train.columns)
    target_col = st.selectbox("Selecione a coluna alvo (y)", cols, index=len(cols)-1 if cols else 0)

st.subheader("2) Base de **Predição** (sem rótulo)")
pred_file = st.file_uploader("Envie um CSV para predição (sem a coluna alvo, opcional)", type=["csv"], key="pred")
df_pred = None
if pred_file is not None:
    try:
        df_pred = pd.read_csv(pred_file)
    except Exception:
        pred_file.seek(0)
        df_pred = pd.read_csv(pred_file, sep=";")
    st.write("Pré-visualização da base de predição:", df_pred.head())

# Treinamento e avaliação
if df_train is not None and target_col is not None:
    # Separar X e y
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    # Identificar tipos
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if num_cols:
        if standardize_numeric:
            transformers.append(("num", StandardScaler(), num_cols))
        else:
            # 'passthrough' mantém números como estão
            transformers.append(("num", "passthrough", num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Escolha do modelo
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=random_state
        )
    else:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])

    # Divisão treino/validação
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Treinar
    pipe.fit(X_train, y_train)

    # Validar
    y_pred = pipe.predict(X_test)

    st.markdown("### ✅ Métricas na base de **validação**")
    average = None
    pos_label = None
    if y_test.dtype == "O" or y_test.nunique() != 2:
        # Multiclasse: usar macro
        average = "macro"
        st.caption("Atenção: Problema multiclasse detectado — as métricas usam **média macro**.")
    else:
        # Binário: escolher rótulo positivo
        unique_labels = sorted(y_test.unique().tolist())
        pos_label = st.selectbox("Escolha o rótulo positivo (para Precisão/Recall)", unique_labels, index=1 if len(unique_labels) > 1 else 0)

    acc = accuracy_score(y_test, y_pred)
    if average is None:
        prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    else:
        prec = precision_score(y_test, y_pred, average=average, zero_division=0)
        rec = recall_score(y_test, y_pred, average=average, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia", f"{acc:.3f}")
    c2.metric("Precisão", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")

    # Matriz de confusão
    st.markdown("### 🧩 Matriz de Confusão")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    fig = plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(values_format='d')
    plt.title("Matriz de Confusão (base de validação)")
    st.pyplot(fig)

    # Predição na base de predição
    st.markdown("---")
    st.markdown("### 🔮 Predições na base **sem rótulo**")
    if df_pred is not None:
        # Verificar colunas compatíveis após preparação
        missing_cols = [c for c in X.columns if c not in df_pred.columns]
        if missing_cols:
            st.warning(f"Atenção: A base de predição não possui {len(missing_cols)} coluna(s) presente(s) no treino: {missing_cols}. Elas serão tratadas como faltantes (One-Hot ignora categorias desconhecidas; numéricos ausentes não são suportados).")

        # Garantir que df_pred tenha apenas colunas do treino (extras serão ignoradas pelo ColumnTransformer)
        X_pred = df_pred.copy()

        # Predizer
        y_pred_deploy = pipe.predict(X_pred)
        proba_available = hasattr(pipe.named_steps["clf"], "predict_proba")
        if proba_available:
            y_proba = pipe.predict_proba(X_pred)
            # Probabilidade da classe positiva (binário) ou score da classe predita (multiclasse)
            if (len(pipe.named_steps["clf"].classes_) == 2):
                # pegar prob da classe escolhida como positiva (se aplicável)
                if pos_label is not None and pos_label in pipe.named_steps["clf"].classes_:
                    pos_index = list(pipe.named_steps["clf"].classes_).index(pos_label)
                else:
                    pos_index = 1 if len(pipe.named_steps["clf"].classes_) > 1 else 0
                score = y_proba[:, pos_index]
                df_out = df_pred.copy()
                df_out["Predição"] = y_pred_deploy
                df_out["Score_Pos"] = score
            else:
                # multiclasse: score da classe predita
                max_scores = y_proba.max(axis=1)
                df_out = df_pred.copy()
                df_out["Predição"] = y_pred_deploy
                df_out["Score_Pred"] = max_scores
        else:
            df_out = df_pred.copy()
            df_out["Predição"] = y_pred_deploy

        st.write("Resultados de predição (primeiras linhas):")
        st.dataframe(df_out.head())

        # Download do CSV completo
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar resultados (.csv)",
            data=csv_bytes,
            file_name="predicoes.csv",
            mime="text/csv"
        )

        # (Opcional) Filtrar apenas positivos (quando binário)
        if df_out["Predição"].nunique() == 2:
            positive_label = st.selectbox("Filtrar apenas predições positivas (opcional) — escolha o rótulo considerado 'positivo':", sorted(df_out["Predição"].unique().tolist()))
            filtered = df_out[df_out["Predição"] == positive_label]
            st.write(f"Registros previstos como **{positive_label}**:", filtered.head())
            csv_pos = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Baixar apenas predições '{positive_label}' (.csv)",
                data=csv_pos,
                file_name=f"predicoes_{positive_label}.csv",
                mime="text/csv"
            )

    else:
        st.info("Envie a base de predição para obter os resultados.")
else:
    st.info("Envie a base de treino/validação e selecione a coluna alvo para continuar.")

st.markdown("---")
st.caption(
    "Dicas: • Acurácia = acertos/total • Precisão = VP/(VP+FP) • Recall = VP/(VP+FN). "
    "Use Precisão quando o custo de FP é alto; use Recall quando o custo de FN é alto."
)
