import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MultiLabelBinarizer
import re
from collections import Counter, defaultdict
import random

st.set_page_config(page_title="Apriori – Treino & Predição", layout="wide")
st.title("🛒 Apriori com Duas Bases: Treino/Teste e Predição")

st.markdown("""
Você é um **consultor de IA** ajudando um **mercado de bairro** a entender padrões de compra para **ofertas**, **layout da loja** e **estoque**.

**Fluxo da atividade**  
1) Carregue **duas bases** (ou gere sintéticas):  
   - **Treino/Teste (1000 transações)** → aprender padrões e **gerar regras**.  
   - **Predição (100 transações)** → **aplicar as regras** para recomendar itens a cada carrinho.  
2) Ajuste os **parâmetros** do Apriori e dos filtros de regras.  
3) Avalie **cobertura** e **qualidade** das recomendações.  
4) Escreva recomendações operacionais para o lojista.
""")

# --------------------------
# Sinthetic data generator
# --------------------------
def gerar_transacoes_sinteticas(n, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    lacteos = ["leite", "queijo", "manteiga", "iogurte"]
    panificados = ["pão", "biscoito", "bolo"]
    cafe_e_afins = ["café", "açúcar", "filtro de café"]
    massas = ["macarrão", "molho de tomate", "queijo ralado"]
    basicos = ["arroz", "feijão", "óleo", "farinha", "sal"]
    frios = ["presunto", "mussarela"]
    limpeza = ["detergente", "esponja", "sabão em pó", "amaciante"]
    higiene = ["papel higiênico", "sabonete", "creme dental", "escova de dente"]
    bebidas = ["refrigerante", "suco", "água mineral"]
    snacks = ["batata chips", "chocolate", "bala"]

    base_pop = lacteos + panificados + cafe_e_afins + massas + basicos + frios + limpeza + higiene + bebidas + snacks

    transacoes = []
    for _ in range(n):
        cesta = set()
        tam = np.random.choice([2,3,4,5,6], p=[0.15,0.30,0.30,0.18,0.07])

        if random.random() < 0.35:
            cesta.add("leite")
            if random.random() < 0.6: cesta.add("pão")
            if random.random() < 0.4: cesta.add("manteiga")
        if random.random() < 0.30:
            cesta.add("café")
            if random.random() < 0.7: cesta.add("açúcar")
            if random.random() < 0.35: cesta.add("filtro de café")
        if random.random() < 0.28:
            cesta.add("macarrão")
            if random.random() < 0.7: cesta.add("molho de tomate")
            if random.random() < 0.5: cesta.add("queijo ralado")
        if random.random() < 0.40:
            cesta.add("arroz")
            if random.random() < 0.75: cesta.add("feijão")
            if random.random() < 0.45: cesta.add("óleo")
        if random.random() < 0.25:
            cesta.add("detergente")
            if random.random() < 0.5: cesta.add("esponja")
        if random.random() < 0.25:
            cesta.add("papel higiênico")
            if random.random() < 0.4: cesta.add("sabonete")

        while len(cesta) < tam:
            cesta.add(random.choice(base_pop))

        transacoes.append(", ".join(sorted(cesta)))

    return pd.DataFrame({"Transacao": transacoes})

# --------------------------
# Helpers
# --------------------------
def split_itens(x: str):
    if pd.isna(x):
        return []
    partes = re.split(r"[;,]", str(x))
    return [p.strip().lower() for p in partes if p and p.strip()]

def listas_para_ohe(series_listas):
    mlb = MultiLabelBinarizer(sparse_output=False)
    ohe = mlb.fit_transform(series_listas)
    df_bin = pd.DataFrame(ohe, columns=mlb.classes_).astype(bool)
    return df_bin, mlb

def fs_to_text(fs):
    return ", ".join(sorted(list(fs)))

def aplicar_regras_em_cesta(itens_cesta_set, regras_df, top_k=None, evitar_existentes=True):
    recs = []
    for _, r in regras_df.iterrows():
        ante = r["antecedents"]
        cons = r["consequents"]
        if ante.issubset(itens_cesta_set):
            sug = [c for c in cons if (not evitar_existentes or c not in itens_cesta_set)]
            if sug:
                recs.append((tuple(sorted(sug)), r["lift"], r["confidence"], r["support"]))
    recs.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    if top_k:
        recs = recs[:top_k]
    return recs

# --------------------------
# Atividade (roteiro didático)
# --------------------------
st.markdown("""
---
## 🎯 Atividade proposta (versão aprimorada)

1) **Prepare os dados**  
   - Use as bases sintéticas (1000 e 100) ou faça **upload** das suas.  
   - Garanta listas de itens na coluna de transações (ex.: `"leite, pão, manteiga"`).

2) **Aprenda as regras** na base **Treino/Teste**  
   - Varie **Suporte**, **Confiança**, **Lift**, **Tamanho** de conjuntos.  
   - Registre **quantas regras** obteve e **quais são as top 5 por *lift***.

3) **Aplique as regras** na base **Predição**  
   - Observe **Cobertura** (% de carrinhos com recomendação) e **Média de recomendações por transação**.  
   - Liste os **Top 10 itens recomendados**.

4) **Recomendações ao negócio**  
   - Escolha **ao menos 2 regras** e explique como usá-las (combos, cross-sell, layout de gôndolas, precificação, *endcaps*).  
   - Proponha **um experimento A/B** simples para validar o impacto de uma recomendação no faturamento.

5) **Reflexões (responder no relatório)**  
   - Como **suporte** impacta a **escalabilidade operacional**?  
   - Como **confiança** e **lift** impactam a **qualidade** das recomendações?  
   - Qual trade-off você encontrou entre **número de regras** e **relevância**?  
   - Se um item é muito popular, como isso influencia o **lift**?

> **Dica**: Foque em regras com **lift > 1** e **confiança** alta; porém, um **suporte** muito baixo pode dificultar a execução prática na loja.
""")
