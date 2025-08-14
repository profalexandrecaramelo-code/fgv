
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AED – Inspeção, Qualidade & Outliers", page_icon="📊", layout="wide")

st.title("📊 Análise Exploratória – Passos 1 e 4 + Qualidade dos Dados")
st.caption("Inspeção inicial, verificação de **qualidade dos dados** (nulos e inconsistências) e **outliers (IQR)**. Gere um **dataset limpo** ao final.")

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

# ========== PASSO 1 – INSPEÇÃO INICIAL ==========
st.divider()
st.header("1) Inspeção inicial")

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

# ========== QUALIDADE DOS DADOS ==========
st.divider()
st.header("Qualidade dos Dados (nulos e inconsistências)")

st.markdown('''
**Regras de validação aplicadas:**  
- `area_m2` > 0  
- `quartos` >= 1 e inteiro  
- `banheiros` >= 1 e inteiro  
- `vagas` >= 0 e inteiro  
- `preco` > 0  
- `ano_construcao` entre 1900 e ano corrente  
- IDs duplicados
''')

# Detect problems per row
current_year = pd.Timestamp.now().year
problems = []

# Prepare duplicate id mask
dup_id_mask = df["id"].duplicated(keep=False) if "id" in df.columns else pd.Series(False, index=df.index)

for i, row in df.iterrows():
    issues = []
    # Missing values
    if row.isna().any():
        issues.append("Valores nulos")
    # Area
    if "area_m2" in df.columns:
        val = row["area_m2"]
        if pd.notna(val) and (val <= 0):
            issues.append("area_m2 inválido (<=0)")
    # Quartos
    if "quartos" in df.columns:
        val = row["quartos"]
        if pd.notna(val):
            if (val < 1) or (float(val) != int(val)):
                issues.append("quartos inválido (>=1 e inteiro)")
    # Banheiros
    if "banheiros" in df.columns:
        val = row["banheiros"]
        if pd.notna(val):
            if (val < 1) or (float(val) != int(val)):
                issues.append("banheiros inválido (>=1 e inteiro)")
    # Vagas
    if "vagas" in df.columns:
        val = row["vagas"]
        if pd.notna(val):
            if (val < 0) or (float(val) != int(val)):
                issues.append("vagas inválido (>=0 e inteiro)")
    # Preço
    if "preco" in df.columns:
        val = row["preco"]
        if pd.notna(val) and (val <= 0):
            issues.append("preco inválido (<=0)")
    # Ano de construção
    if "ano_construcao" in df.columns:
        val = row["ano_construcao"]
        if pd.notna(val) and not (1900 <= int(val) <= current_year):
            issues.append("ano_construcao inválido (1900..ano_corrente)")
    # Duplicidade de ID
    if dup_id_mask.iloc[i]:
        issues.append("id duplicado")
    problems.append("; ".join(issues))

df["_problemas"] = problems
df["_tem_problema"] = df["_problemas"].str.len() > 0

st.subheader("Linhas com problemas de qualidade")
df_prob = df[df["_tem_problema"]].copy()
st.write(f"Total com problemas: **{df_prob.shape[0]}**")
if df_prob.empty:
    st.success("Nenhum problema encontrado com base nas regras definidas.")
else:
    st.dataframe(df_prob, use_container_width=True)

# ========== PASSO 4 – OUTLIERS (IQR) ==========
st.divider()
st.header("4) Identificação de Outliers (Método IQR)")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) == 0:
    st.warning("Não há colunas numéricas para análise de outliers.")
    st.stop()

sel_cols = st.multiselect(
    "Selecione colunas numéricas para avaliar outliers",
    options=num_cols,
    default=[c for c in num_cols if c not in ["id"]][:3]
)

iqr_multiplier = st.slider("Fator multiplicador do IQR (padrão=1.5)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

def iqr_bounds(s, k=1.5):
    s = s.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k*iqr
    upper = q3 + k*iqr
    return lower, upper, q1, q3, iqr, s.median()

if sel_cols:
    outlier_mask = pd.Series(False, index=df.index)
    bounds_info = {}
    for col in sel_cols:
        valid = df[col].dropna()
        if valid.empty:
            continue
        lower, upper, q1, q3, iqr, med = iqr_bounds(valid, k=iqr_multiplier)
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_mask |= mask.fillna(False)
        bounds_info[col] = {"lower": lower, "upper": upper, "Q1": q1, "Q3": q3, "IQR": iqr, "mediana": med}

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

    st.subheader("Outliers (qualquer coluna selecionada)")
    out_df = df.loc[outlier_mask].copy()
    st.write(f"Total de outliers: **{out_df.shape[0]}**")
    st.dataframe(out_df, use_container_width=True)

    st.subheader("🖼️ Explicação visual do IQR (por coluna)")
    col_visual = st.selectbox(
        "Escolha uma coluna numérica para visualizar o IQR, quartis e limites de outliers",
        options=[c for c in num_cols if c != "id"]
    )
    if col_visual:
        series = df[col_visual].dropna()
        lower, upper, q1, q3, iqr, med = iqr_bounds(series, k=iqr_multiplier)

        # BOX PLOT
        st.subheader("Boxplot com limites do IQR")
        fig1, ax1 = plt.subplots()
        ax1.boxplot(series, vert=False, showmeans=True)
        ax1.axvline(lower, linestyle="--")
        ax1.axvline(upper, linestyle="--")
        ax1.axvline(q1, linestyle=":")
        ax1.axvline(q3, linestyle=":")
        ax1.axvline(med, linestyle="-")
        ax1.set_xlabel(col_visual)
        ax1.set_title(f"Boxplot de {col_visual} | Q1={q1:.2f}, Mediana={med:.2f}, Q3={q3:.2f}")
        st.pyplot(fig1)

        # HISTOGRAM
        st.subheader("Histograma com marcações do IQR")
        fig2, ax2 = plt.subplots()
        ax2.hist(series, bins=30)
        ax2.axvline(lower, linestyle="--", label="Limite Inferior")
        ax2.axvline(upper, linestyle="--", label="Limite Superior")
        ax2.axvline(q1, linestyle=":", label="Q1")
        ax2.axvline(q3, linestyle=":", label="Q3")
        ax2.axvline(med, linestyle="-", label="Mediana")
        ax2.set_xlabel(col_visual)
        ax2.set_ylabel("Frequência")
        ax2.set_title(f"Histograma de {col_visual} com IQR")
        ax2.legend()
        st.pyplot(fig2)

# ========== FUSÃO DE STATUS & DOWNLOAD DA BASE LIMPA ==========
st.divider()
st.header("Gerar base limpa")

st.markdown('''
- 🔴 **Problema de qualidade** (nulos/inconsistências)  
- 🟡 **Outlier** (IQR nas colunas selecionadas)  
- 🟣 **Ambos**  
- 🟢 **OK**
''')

# recompute outlier mask using current selection (if none, all False)
if sel_cols:
    final_outlier_mask = pd.Series(False, index=df.index)
    for col in sel_cols:
        valid = df[col].dropna()
        if valid.empty:
            continue
        lower, upper, q1, q3, iqr, med = iqr_bounds(valid, k=iqr_multiplier)
        final_outlier_mask |= ((df[col] < lower) | (df[col] > upper)).fillna(False)
else:
    final_outlier_mask = pd.Series(False, index=df.index)

status = np.where(df["_tem_problema"] & final_outlier_mask, "🟣 problema + outlier",
         np.where(df["_tem_problema"], "🔴 problema",
         np.where(final_outlier_mask, "🟡 outlier", "🟢 ok")))

df_preview = df.copy()
df_preview["_status"] = status

# Color preview
def highlight_row(row):
    color = ""
    if row["_status"].startswith("🟣"):
        color = "background-color: #e8ddff"
    elif row["_status"].startswith("🔴"):
        color = "background-color: #ffe5e5"
    elif row["_status"].startswith("🟡"):
        color = "background-color: #fff7d6"
    elif row["_status"].startswith("🟢"):
        color = "background-color: #e8f5e9"
    return [color] * len(row)

st.subheader("Prévia com status (cores)")
st.dataframe(df_preview.head(100).style.apply(highlight_row, axis=1), use_container_width=True)

clean_df = df.loc[~df["_tem_problema"] & ~final_outlier_mask].copy()
st.write(f"Registros originais: **{df.shape[0]}** | Removidos: **{(df.shape[0]-clean_df.shape[0])}** | Restantes (limpos): **{clean_df.shape[0]}**")

def to_csv_bytes(d):
    return d.drop(columns=["_tem_problema"], errors="ignore").to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Baixar BASE LIMPA (.csv)",
    data=to_csv_bytes(clean_df),
    file_name="base_limpa.csv",
    mime="text/csv",
    use_container_width=True
)
