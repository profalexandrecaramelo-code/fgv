# streamlit_app.py (tarefa1_supervisionado_v3)
# ------------------------------------------------------
# Exercício: Avaliação Executiva de um Sistema de IA (Supervisionado)
# Requisitos do professor:
# 1) Apresentar um problema de negócio.
# 2) Upload de base histórica e split 70/30 (treino/teste).
# 3) O sistema usa IA SUPERVISIONADA e resolve PARCIALMENTE o problema.
# 4) Exibir APENAS a ACURÁCIA.
# 5) Exibir a base utilizada (mesmo com erros) e DESTACAR erros por cor.
# 6) Permitir inserir UMA NOVA BASE (sem alvo) para obter as PREDIÇÕES do sistema.
# 7) Mostrar os pedidos com risco de atraso (predição = 1).
# 8) As equipes analisam e propõem ações do EXECUTIVO com base nos 10 passos.
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

# ===============================
# 1) Introdução
# ===============================
st.title("📦 Avaliação Executiva de um Sistema de IA Supervisionado")
st.markdown("""
Este exercício simula o uso de **IA supervisionada** para prever **atrasos em entregas**.
O objetivo é que o **Executivo** compreenda limitações, riscos e decisões necessárias em todas as etapas.
""")

# ===============================
# 2) Upload da base histórica
# ===============================
st.header("1) Upload da Base Histórica")
file = st.file_uploader("📥 Envie a base histórica (CSV com coluna alvo `atraso`)", type=["csv"], key="hist")

if file is not None:
    df = pd.read_csv(file)

    st.write("Prévia da base histórica:")
    st.dataframe(df.head(), use_container_width=True)

    # Destacar erros (valores nulos ou inconsistentes)
    erros = df.isna() | df.applymap(lambda x: isinstance(x, str) and not x.isnumeric())
    st.write("Linhas destacadas em vermelho possuem potenciais erros:")
    st.dataframe(df.style.apply(lambda row: ['background-color: red' if e else '' for e in row], axis=1))

    # ===============================
    # 3) Treino/Teste e modelo
    # ===============================
    st.header("2) Treinamento e Avaliação")

    if "atraso" in df.columns:
        X = df.drop(columns=["atraso"])
        y = df["atraso"]

        # Identificar colunas numéricas e categóricas
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X.select_dtypes(include=["object"]).columns

        # Pré-processamento
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), num_cols),
                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols)
            ]
        )

        # Modelo
        pipe = Pipeline(steps=[("preprocessor", preprocessor),
                               ("classifier", LogisticRegression(max_iter=1000))])

        # Split 70/30
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Acurácia
        acc = accuracy_score(y_test, y_pred)
        st.metric("Acurácia do Modelo", f"{acc:.2%}")

        # ===============================
        # 4) Nova base para predição
        # ===============================
        st.header("3) Predição com Nova Base")
        st.caption("Envie um CSV **sem a coluna alvo** para obter as **predições** do modelo treinado acima.")
        new_file = st.file_uploader(
            "📥 Envie um CSV para predição (mesmas colunas de entrada, sem a classe)", 
            type=["csv"], key="novo"
        )

        if new_file is not None:
            try:
                df_new = pd.read_csv(new_file)
            except Exception:
                new_file.seek(0)
                df_new = pd.read_csv(new_file, sep=';')

            st.write("Prévia da nova base:")
            st.dataframe(df_new.head(), use_container_width=True)

            try:
                preds_new = pipe.predict(df_new)
                out = df_new.copy()
                out["predicao_atraso"] = preds_new

                # 🔎 Mostrar apenas pedidos com risco de atraso
                atrasos = out[out["predicao_atraso"] == 1]
                st.success(f"Foram identificados {len(atrasos)} pedidos com risco de atraso.")
                st.dataframe(atrasos, use_container_width=True)

                # Botão para baixar todas as predições
                st.download_button(
                    "⬇️ Baixar todas as predições (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predicoes_nova_base.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.warning(f"Não foi possível prever com a nova base: {e}")

# ===============================
# 5) Discussão Executiva
# ===============================
st.markdown("---")
st.header("4) Discussão em Equipe — Ações do Executivo")
st.markdown(
    """
    1. **Objetivos** — O sistema ajuda a atingir a meta de negócio?  
    2. **Fontes de dados** — Há fontes críticas faltando?  
    3. **Refinamento** — Os erros destacados (faltantes, duplicados, tipos) comprometem decisões?  
    4. **Variáveis** — Quais atributos devem ser **exigidos** ou criados?  
    5. **Restrições** — Há requisitos de explicabilidade, tempo de resposta ou custo a reforçar antes do próximo ciclo?  
    6. **Aprendizado** — O tipo (supervisionado) é adequado? Precisamos rotular melhor os dados (definições claras de atraso)?  
    7. **Algoritmo** — Mesmo exibindo só acurácia, precisamos autorizar testes com alternativas mais explicáveis/robustas?  
    8. **Treinamento** — O 70/30 está ok? Precisamos de política de versão de modelos e dados?  
    9. **Avaliação** — Só acurácia basta para o risco?  
    10. **Implantação/Monitoramento** — Se fosse para produção, que SLAs e auditorias o executivo cobraria?  
    """
)
st.success("Objetivo pedagógico: evidenciar que **o executivo decide rumos e políticas em TODAS as etapas, não apenas ao final.**")
