
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AED - Análise Exploratória de Dados", page_icon="📊", layout="wide")
st.title("📊 Atividade em Equipe - Análise Exploratória de Dados")
st.caption("Carregue um CSV ou use a base fornecida. Identifique problemas de **Completude**, **Consistência**, **Unicidade** e **Outliers (IQR)**. Gere uma **base limpa**.")

@st.cache_data
def load_sample():
    return pd.read_csv("imoveis.csv")

uploaded = st.file_uploader("Envie um CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ Dataset carregado do upload.")
else:
    st.info("Use a base `imoveis.csv` incluída no app.")
    try:
        df = load_sample()
    except Exception:
        st.error("A base não foi encontrada. Faça upload de um CSV.")
        st.stop()

# ---------- PASSO 1
st.divider()
st.header("1) Inspeção inicial")
c1,c2,c3 = st.columns(3)
with c1: st.metric("Linhas (n)", f"{df.shape[0]:,}".replace(",","."))
with c2: st.metric("Colunas (p)", df.shape[1])
with c3: st.metric("% Nulos (média)", f"{df.isna().mean().mean()*100:.2f}%")
st.dataframe(df.head(12), use_container_width=True)
with st.expander("Tipos & nulos por coluna"):
    info_df = pd.DataFrame({"dtype": df.dtypes.astype(str), "n_nulos": df.isna().sum(), "%_nulos": (df.isna().mean()*100).round(2)})
    st.dataframe(info_df, use_container_width=True)

# ---------- QUALIDADE: 3Cs
st.divider()
st.header("Qualidade dos Dados (3Cs)")

current_year = pd.Timestamp.now().year
probs = []
cats = []

# Unicidade
dup_id_mask = df["id"].duplicated(keep=False) if "id" in df.columns else pd.Series(False, index=df.index)
# Duplicidade de linha (todas as colunas iguais) - marca as duplicadas (mantém a 1ª)
dup_row_mask = df.duplicated(keep="first")

def categorias_linha(row, idx):
    cat = []
    # Completude
    if row.isna().any(): cat.append("Completude: nulos")
    # Consistência (regras cruzadas e de domínio)
    # domínios básicos
    if "area_m2" in df.columns and pd.notna(row["area_m2"]) and row["area_m2"] <= 0: cat.append("Consistência: area_m2 <= 0")
    if "quartos" in df.columns and pd.notna(row["quartos"]) and row["quartos"] < 1: cat.append("Consistência: quartos < 1")
    if "banheiros" in df.columns and pd.notna(row["banheiros"]) and row["banheiros"] < 1: cat.append("Consistência: banheiros < 1")
    if "vagas" in df.columns and pd.notna(row["vagas"]) and row["vagas"] < 0: cat.append("Consistência: vagas < 0")
    if "preco" in df.columns and pd.notna(row["preco"]) and row["preco"] <= 0: cat.append("Consistência: preco <= 0")
    if "ano_construcao" in df.columns and pd.notna(row["ano_construcao"]) and not (1900 <= int(row["ano_construcao"]) <= current_year):
        cat.append("Consistência: ano_construcao fora do intervalo")
    # regras cruzadas
    if all(c in df.columns for c in ["suites","quartos"]) and pd.notna(row["suites"]) and pd.notna(row["quartos"]) and row["suites"] > row["quartos"]:
        cat.append("Consistência: suites > quartos")
    if all(c in df.columns for c in ["banheiros","quartos"]) and pd.notna(row["banheiros"]) and pd.notna(row["quartos"]) and row["banheiros"] > row["quartos"] + 2:
        cat.append("Consistência: banheiros > quartos + 2")
    if all(c in df.columns for c in ["vagas","quartos"]) and pd.notna(row["vagas"]) and pd.notna(row["quartos"]) and row["vagas"] > row["quartos"] + 3:
        cat.append("Consistência: vagas > quartos + 3")
    if all(c in df.columns for c in ["tipo","area_m2"]) and pd.notna(row["tipo"]) and pd.notna(row["area_m2"]) and (row["tipo"]=="Cobertura") and (row["area_m2"] < 90):
        cat.append("Consistência: cobertura com área < 90m²")
    if all(c in df.columns for c in ["preco","area_m2"]) and pd.notna(row["preco"]) and pd.notna(row["area_m2"]) and row["area_m2"]>0:
        pm2 = row["preco"]/row["area_m2"]
        if pm2 < 1500: cat.append("Consistência: preço/m² irrealmente baixo")
        if pm2 > 50000: cat.append("Consistência: preço/m² irrealmente alto")
    # Unicidade
    if dup_id_mask.iloc[idx]: cat.append("Unicidade: id duplicado")
    if dup_row_mask.iloc[idx]: cat.append("Unicidade: linha duplicada")
    return "; ".join(cat)

cats = [categorias_linha(df.iloc[i], i) for i in range(df.shape[0])]
df["_categorias_problema"] = cats
df["_tem_problema_3Cs"] = df["_categorias_problema"].str.len() > 0

st.subheader("Resumo (contagem por tipo de problema)")
if df["_tem_problema_3Cs"].any():
    all_cats = []
    for c in df["_categorias_problema"]:
        if c:
            all_cats.extend([x.strip() for x in c.split(";")])
    summary = pd.Series(all_cats).value_counts().rename_axis("categoria").reset_index(name="qtd")
    st.dataframe(summary, use_container_width=True)
else:
    st.success("Nenhum problema de Completude/Consistência/Unicidade foi encontrado.")

st.subheader("Linhas com problemas (3Cs)")
st.dataframe(df[df["_tem_problema_3Cs"]], use_container_width=True)

# ---------- OUTLIERS (IQR)
st.divider()
st.header("4) Outliers pelo IQR")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
sel_cols = st.multiselect("Selecione colunas numéricas", options=[c for c in num_cols if c not in ["id"]], default=[c for c in num_cols if c not in ["id"]][:3])
k = st.slider("Fator do IQR (padrão=1.5)", 0.5, 3.0, 1.5, 0.1)

def iqr_bounds(s, kk=1.5):
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - kk*iqr, q3 + kk*iqr, q1, q3, iqr

if sel_cols:
    outlier_mask = pd.Series(False, index=df.index)
    for col in sel_cols:
        s = df[col].dropna()
        if s.empty: 
            continue
        lo, hi, *_ = iqr_bounds(s, kk=k)
        outlier_mask |= ((df[col] < lo) | (df[col] > hi)).fillna(False)
    st.write(f"Total de outliers (qualquer coluna): **{int(outlier_mask.sum())}**")
    st.dataframe(df[outlier_mask], use_container_width=True)
else:
    outlier_mask = pd.Series(False, index=df.index)
    st.info("Selecione pelo menos 1 coluna.")

# ---------- GERAÇÃO BASE LIMPA
st.divider()
st.header("Gerar base limpa")

st.markdown('''
- 🔴 **Problema (3Cs)**  
- 🟡 **Outlier (IQR)**  
- 🟣 **Ambos**  
- 🟢 **OK**
''')

status = np.where(df["_tem_problema_3Cs"] & outlier_mask, "🟣 problema + outlier",
         np.where(df["_tem_problema_3Cs"], "🔴 problema",
         np.where(outlier_mask, "🟡 outlier", "🟢 ok")))
df_prev = df.copy()
df_prev["_status"] = status

def highlight_row(row):
    if row["_status"].startswith("🟣"): return ["background-color: #e8ddff"]*len(row)
    if row["_status"].startswith("🔴"): return ["background-color: #ffe5e5"]*len(row)
    if row["_status"].startswith("🟡"): return ["background-color: #fff7d6"]*len(row)
    if row["_status"].startswith("🟢"): return ["background-color: #e8f5e9"]*len(row)
    return [""]*len(row)

st.subheader("Prévia com status (cores)")
st.dataframe(df_prev.head(120).style.apply(highlight_row, axis=1), use_container_width=True)

clean_df = df.loc[~df["_tem_problema_3Cs"] & ~outlier_mask].drop(columns=["_tem_problema_3Cs"], errors="ignore").copy()
st.write(f"Registros originais: **{df.shape[0]}** | Removidos: **{(df.shape[0]-clean_df.shape[0])}** | Restantes (limpos): **{clean_df.shape[0]}**")

def to_csv_bytes(d):
    return d.to_csv(index=False).encode("utf-8")

st.download_button("⬇️ Baixar BASE LIMPA (.csv)", data=to_csv_bytes(clean_df), file_name="base_limpa.csv", mime="text/csv", use_container_width=True)
