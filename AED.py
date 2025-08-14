
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="AED – Inspeção & Outliers", page_icon="📊", layout="wide")

st.title("📊 Análise Exploratória – Passos 1 e 4 (Inspeção & Outliers)")
st.caption("Envie um CSV ou use a base fictícia de imóveis. O foco é **inspeção inicial** e **identificação de outliers (IQR)**.")

# --- Data loader
@st.cache_data
def load_sample():
    return pd.read_csv("imoveis.csv")

uploaded = st.file_uploader("Envie um arquivo CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ Dataset carregado do upload.")
else:
    st.info("Usando a base fictícia `imoveis.csv` incluída no app.")
    try:
        df = load_sample()
    except Exception as e:
        st.error("A base fictícia não foi encontrada no ambiente. Faça upload de um CSV.")
        st.stop()

st.divider()
st.header("1) Inspeção inicial")

# Overview
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Linhas (n)", f"{df.shape[0]:,}".replace(",", "."))
with c2:
    st.metric("Colunas (p)", df.shape[1])
with c3:
    st.metric("Percentual de Nulos", f"{df.isna().mean().mean()*100:.2f}%")

st.subheader("Amostra das primeiras linhas")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("Tipos de dados & nulos por coluna", expanded=False):
    info_df = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_nulos": df.isna().sum(),
        "%_nulos": (df.isna().mean()*100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)

with st.expander("Estatística descritiva (numéricas)", expanded=False):
    st.dataframe(df.select_dtypes(include=np.number).describe().T, use_container_width=True)

st.divider()
st.header("4) Identificação de Outliers (Método IQR)")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) == 0:
    st.warning("Não há colunas numéricas para análise de outliers.")
    st.stop()

sel_cols = st.multiselect("Selecione colunas numéricas para avaliar outliers", options=num_cols, default=[c for c in num_cols if c not in ["id"]][:3])

iqr_multiplier = st.slider("Fator multiplicador do IQR (padrão=1.5)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)


st.markdown("### 📖 O que é IQR (Intervalo Interquartílico)")
st.write(
    "O **IQR** (Interquartile Range) mede a variação dos 50% centrais dos dados, calculado como `Q3 - Q1`. "
    "Valores muito abaixo de `Q1 - k×IQR` ou muito acima de `Q3 + k×IQR` são considerados **outliers**. "
    "O fator `k` geralmente é 1.5."
)

import matplotlib.pyplot as plt

if sel_cols:
    col_plot = sel_cols[0]  # mostrar para a primeira coluna selecionada
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.boxplot(df[col_plot].dropna(), vert=False)
    ax.set_title(f"Boxplot de {col_plot} (com outliers)")
    st.pyplot(fig)

def iqr_bounds(s, k=1.5):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k*iqr
    upper = q3 + k*iqr
    return lower, upper, q1, q3, iqr

if sel_cols:
    outlier_mask = pd.Series(False, index=df.index)
    bounds_info = {}
    for col in sel_cols:
        valid = df[col].dropna()
        if valid.empty:
            continue
        lower, upper, q1, q3, iqr = iqr_bounds(valid, k=iqr_multiplier)
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_mask |= mask.fillna(False)
        bounds_info[col] = {"lower": lower, "upper": upper, "Q1": q1, "Q3": q3, "IQR": iqr}

    st.subheader("Resumo por coluna (limites IQR)")
    summary = []
    for col in sel_cols:
        b = bounds_info[col]
        n_out = int(((df[col] < b["lower"]) | (df[col] > b["upper"])).sum())
        summary.append({
            "coluna": col,
            "Q1": b["Q1"],
            "Q3": b["Q3"],
            "IQR": b["IQR"],
            "limite_inferior": b["lower"],
            "limite_superior": b["upper"],
            "n_outliers": n_out
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    st.subheader("Linhas identificadas como outliers (qualquer coluna selecionada)")
    out_df = df.loc[outlier_mask].copy()
    st.write(f"Total de outliers: **{out_df.shape[0]}**")
    st.dataframe(out_df, use_container_width=True)

    st.subheader("Dados sem outliers (com base nas colunas selecionadas)")
    clean_df = df.loc[~outlier_mask].copy()
    st.write(f"Total sem outliers: **{clean_df.shape[0]}**")
    st.dataframe(clean_df.head(15), use_container_width=True)

    # Downloads
    def to_csv_bytes(d):
        return d.to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Baixar OUTLIERS (.csv)", data=to_csv_bytes(out_df), file_name="outliers.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("⬇️ Baixar DADOS SEM OUTLIERS (.csv)", data=to_csv_bytes(clean_df), file_name="dados_sem_outliers.csv", mime="text/csv", use_container_width=True)

    st.caption("Dica: ajuste o multiplicador do IQR para tornar a regra mais rígida (menor) ou mais permissiva (maior).")

else:
    st.info("Selecione ao menos uma coluna para iniciar a análise de outliers.")
