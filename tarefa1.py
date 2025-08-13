# streamlit_app.py
# ------------------------------------------------------
# Exercício: Avaliação Executiva de um Sistema de IA (Supervisionado)
# Requisitos do professor:
# 1) Apresentar um problema de negócio.
# 2) Upload de base histórica e split 70/30 (treino/teste).
# 3) O sistema usa IA supervisionada e resolve PARCIALMENTE o problema.
# 4) Exibir APENAS a ACURÁCIA.
# 5) Exibir a base utilizada (mesmo com erros).
# 6) Apontar ONDE há erros na base, colorindo as células problemáticas.
# As equipes analisam e propõem ações do EXECUTIVO com base nos 10 passos.
# ------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Exercício — Avaliação Executiva de IA (70/30)", layout="wide")
st.title("Exercício — Avaliação Executiva de um Sistema de IA (70/30)")

# 1) Problema de negócio (exemplo claro para as equipes)
st.markdown(
    """
    ### 🧩 Problema de Negócio (Exemplo)
    A empresa **Entrega Rápida** sofre com **atrasos nas entregas** e quer **priorizar pedidos** com maior risco de atraso.
    Seu papel é **avaliar um sistema de IA supervisionado** (prototipado) que **resolve parcialmente** o problema e decidir **ações executivas**.

    **O que o sistema faz:** treina um modelo simples em 70% dos dados e mede **apenas a acurácia** em 30% dos dados.
    **O que cabe à equipe:** interpretar o resultado, analisar a qualidade da base (com erros) e propor decisões executivas.
    """
)

st.markdown("---")

# 2) Upload e seleção de coluna alvo
st.header("1) Upload da Base e Seleção da Classe")
file = st.file_uploader("📥 Envie um CSV (delimitador vírgula)", type=["csv"]) 
random_state = st.sidebar.number_input("Semente (random_state)", min_value=0, value=42, step=1)

if file is None:
    st.info("Envie um arquivo .csv contendo a coluna alvo (ex.: target, label, classe). Mantenha os erros: o objetivo é avaliá-los.")
    st.stop()

# Tenta ler separado por vírgula; se falhar, tenta ponto e vírgula
try:
    df_raw = pd.read_csv(file)
except Exception:
    file.seek(0)
    df_raw = pd.read_csv(file, sep=';')

# 5) Exibir a base (mesmo com erros) + 6) Destacar erros por cor
st.header("2) Base Utilizada e Erros Destacados")

# Heurísticas simples de detecção de problemas (células):
# - Valores faltantes (NaN)
# - Linhas duplicadas (marca a linha toda)
# - Tipagem inconsistente em colunas potencialmente numéricas (não-conversíveis)
# - Valores fora de faixa numérica opcional (z-score > 4) como possível outlier (fraco indicativo de erro)

# Duplicados (boolean mask por linha)
dup_mask = df_raw.duplicated(keep=False)

# Tentar identificar colunas "numericáveis"
convertible_numeric = []
non_numeric_cells = pd.DataFrame(False, index=df_raw.index, columns=df_raw.columns)
for c in df_raw.columns:
    # Tenta converter e vê quantos viram NaN a mais do que já eram NaN
    try:
        coerced = pd.to_numeric(df_raw[c], errors='coerce')
        # Marca como potencialmente numérica se pelo menos metade converteu
        if coerced.notna().mean() >= 0.5:
            convertible_numeric.append(c)
            # Células originalmente não numéricas que viraram NaN diferem de NaN original
            non_numeric_cells[c] = coerced.isna() & (~df_raw[c].isna())
    except Exception:
        pass

# Possíveis outliers por z-score > 4 apenas nas colunas convertible_numeric
outlier_cells = pd.DataFrame(False, index=df_raw.index, columns=df_raw.columns)
for c in convertible_numeric:
    coerced = pd.to_numeric(df_raw[c], errors='coerce')
    m = coerced.mean()
    s = coerced.std(ddof=0)
    if s and s > 0:
        z = (coerced - m).abs() / s
        outlier_cells[c] = z > 4

# Máscara de faltantes
na_cells = df_raw.isna()

# Função de estilo por célula
def style_errors(val, row_idx, col_name):
    styles = []
    # Ordem de prioridade: duplicado (linha), faltante, não-numérico indevido, outlier
    if dup_mask.iloc[row_idx]:
        styles.append("background-color: #ffe0e0")  # vermelho claro para duplicados (linha inteira)
    if pd.isna(val):
        styles.append("background-color: #fff3cd")  # amarelo claro para NaN
    if non_numeric_cells.loc[row_idx, col_name]:
        styles.append("background-color: #e0f7ff")  # azul claro para tipagem inconsistente
    if outlier_cells.loc[row_idx, col_name]:
        styles.append("background-color: #e6ffe6")  # verde claro para outlier
    return ";".join(styles) if styles else ""

# Aplica Styler célula a célula
styled = df_raw.style.format(precision=3)
for r in range(len(df_raw)):
    for c in df_raw.columns:
        styled = styled.set_properties(subset=pd.IndexSlice[r, c], **{"background-color": None})

styled = styled.apply(lambda s: [style_errors(v, s.index[i], s.name) for i, v in enumerate(s)], axis=0)

st.caption("Cores: vermelho=duplicado (linha), amarelo=faltante, azul=tipagem inconsistente, verde=possível outlier.")
st.dataframe(styled, use_container_width=True)

# 3) IA supervisionada simples (parcial): Logistic Regression
st.header("3) Treino 70% / Teste 30% — IA Supervisionada (Protótipo Parcial)")

# Seleção da coluna alvo (tentativa automática)
def infer_target(df: pd.DataFrame):
    for cand in ["target", "label", "classe", "y"]:
        if cand in df.columns:
            return cand
    return None

target_guess = infer_target(df_raw)
cols = ["— selecione —"] + df_raw.columns.tolist()
sel_index = cols.index(target_guess) if target_guess in df_raw.columns else 0

label_col = st.selectbox("Coluna alvo (classe)", options=cols, index=sel_index)
if label_col == "— selecione —":
    st.warning("Selecione a coluna alvo para continuar.")
    st.stop()

# Define X, y e split 70/30
X = df_raw.drop(columns=[label_col])
y = df_raw[label_col]

# Identificação de tipos
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.columns.difference(num_cols).tolist()

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline(steps=[("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ]
)

model = LogisticRegression(max_iter=300)
pipe = Pipeline(steps=[("pre", pre), ("clf", model)])

# Estratifica se for classificação com poucas classes
strat = y if y.nunique() <= 20 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state, stratify=strat)

# Treina e mede APENAS acurácia
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
acc = accuracy_score(y_test, preds)

col1, col2 = st.columns([1,2])
with col1:
    st.metric("Acurácia (30% teste)", f"{acc:.3f}")
with col2:
    st.info("Conforme o enunciado do exercício, **somente a acurácia** é exibida. As equipes devem discutir se esse número, sozinho, é suficiente para a decisão.")

st.markdown("---")

# 6) Guia de discussão focado no papel do EXECUTIVO pelos 10 passos
st.header("4) Discussão em Equipe — Ações do Executivo (10 Passos)")
st.markdown(
    """
    1. **Objetivos** — A acurácia apresentada ajuda a atingir a meta de negócio? O que falta medir (ex.: recall de atrasos)?
    2. **Fontes de dados** — Há fontes críticas faltando (ex.: clima em tempo real)? O executivo pode habilitar acesso?
    3. **Refinamento** — Os erros destacados (faltantes, duplicados, tipos) comprometem decisões? Quais políticas de qualidade aprovar?
    4. **Variáveis** — Quais atributos devem ser **exigidos** ou criados (ex.: densidade de paradas, janela de despacho)?
    5. **Restrições** — Há requisitos de explicabilidade, tempo de resposta ou custo a reforçar antes do próximo ciclo?
    6. **Aprendizado** — O tipo (supervisionado) é adequado? Precisamos rotular melhor os dados (definições claras de atraso)?
    7. **Algoritmo** — Mesmo exibindo só acurácia, precisamos autorizar testes com alternativas mais explicáveis/robustas?
    8. **Treinamento** — O 70/30 está ok? Precisamos de política de versão de modelos e dados?
    9. **Avaliação** — Só acurácia basta para o risco? Que **métrica mandatória** o executivo exige no próximo ciclo?
    10. **Implantação/Monitoramento** — Se fosse para produção, que SLAs e auditorias o executivo cobraria?
    """
)

st.success("Objetivo pedagógico: evidenciar que **o executivo decide rumos e políticas** em TODAS as etapas, não apenas ao final.")
