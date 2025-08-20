
import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MultiLabelBinarizer
import re
from collections import Counter
import random

# --------------------------
# Config & Header
# --------------------------
st.set_page_config(page_title="Apriori – Treino & Predição", layout="wide")
st.title("🛒 Apriori com Duas Bases: Treino/Teste e Predição")

st.markdown('''
Você é um **consultor de IA** ajudando um **mercado de bairro** a entender padrões de compra para **ofertas**, **layout da loja** e **estoque**.

**Fluxo da atividade**  
1) Carregue **duas bases** (ou gere sintéticas):  
   - **Treino/Teste (1000 transações)** → aprender padrões e **gerar regras**.  
   - **Predição (100 transações)** → **aplicar as regras** para **recomendar itens por carrinho**.  
2) Ajuste os **parâmetros** do Apriori e dos filtros de regras.  
3) Avalie **cobertura** e **qualidade** das recomendações** e proponha ações ao lojista.
''')

# --------------------------
# Gerador de bases sintéticas
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

    base_pop = (
        lacteos + panificados + cafe_e_afins + massas + basicos +
        frios + limpeza + higiene + bebidas + snacks
    )

    transacoes = []
    for _ in range(n):
        cesta = set()
        tam = np.random.choice([2, 3, 4, 5, 6], p=[0.15, 0.30, 0.30, 0.18, 0.07])

        # Padrões (correlações) para tornar Apriori significativo
        if random.random() < 0.35:
            cesta.add("leite")
            if random.random() < 0.60: cesta.add("pão")
            if random.random() < 0.40: cesta.add("manteiga")
        if random.random() < 0.30:
            cesta.add("café")
            if random.random() < 0.70: cesta.add("açúcar")
            if random.random() < 0.35: cesta.add("filtro de café")
        if random.random() < 0.28:
            cesta.add("macarrão")
            if random.random() < 0.70: cesta.add("molho de tomate")
            if random.random() < 0.50: cesta.add("queijo ralado")
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

        # Completa a cesta até o tamanho alvo
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

def carregar_csv(arquivo):
    # tenta ler com vírgula; se falhar, usa ponto e vírgula
    try:
        df = pd.read_csv(arquivo)
    except Exception:
        arquivo.seek(0)
        df = pd.read_csv(arquivo, sep=";")
    return df

def preparar_ohe(df, col_transacoes):
    listas = df[col_transacoes].apply(split_itens)
    df_bin, mlb = listas_para_ohe(listas)
    return df_bin, mlb, listas

# Aplicação das regras em um carrinho
def aplicar_regras_em_cesta(itens_cesta_set, regras_df, top_k=None):
    recs = []
    for _, r in regras_df.iterrows():
        ante = r["antecedents"]
        cons = r["consequents"]
        if ante.issubset(itens_cesta_set):
            sugeridos = [c for c in cons if c not in itens_cesta_set]
            if sugeridos:
                recs.append((tuple(sorted(sugeridos)), r["lift"], r["confidence"], r["support"]))
    recs.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    if top_k:
        recs = recs[:top_k]
    return recs

# --------------------------
# Geração/Download de exemplos
# --------------------------
col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("🧪 Gerar base **Treino/Teste** (1000)"):
        df_tt = gerar_transacoes_sinteticas(1000, seed=42)
        st.download_button("📥 Baixar Treino/Teste (1000)", df_tt.to_csv(index=False).encode("utf-8"),
                           "treino_1000.csv", "text/csv")
with col_b:
    if st.button("🧪 Gerar base **Predição** (100)"):
        df_pr = gerar_transacoes_sinteticas(100, seed=777)
        st.download_button("📥 Baixar Predição (100)", df_pr.to_csv(index=False).encode("utf-8"),
                           "predicao_100.csv", "text/csv")
with col_c:
    st.info("Use os botões para baixar exemplos. Depois faça o upload das suas bases.")

# --------------------------
# Upload das bases
# --------------------------
st.markdown("### Upload das Bases")
col1, col2 = st.columns(2)
with col1:
    up_tt = st.file_uploader("🔼 Base **Treino/Teste** (CSV, ≥ 1000 linhas)", type=["csv"], key="tt")
with col2:
    up_pr = st.file_uploader("🔼 Base **Predição** (CSV, ≥ 100 linhas)", type=["csv"], key="pr")

# --------------------------
# Parâmetros
# --------------------------
st.sidebar.header("Parâmetros do Apriori e Regras")
min_sup = st.sidebar.slider("Suporte mínimo (%)", 1, 50, 8,
                            help="Percentual mínimo de cestas em que o itemset ocorre.") / 100
max_len = st.sidebar.slider("Tamanho máximo do itemset", 1, 5, 3)
min_conf = st.sidebar.slider("Confiança mínima (%)", 10, 100, 50) / 100
min_lift = st.sidebar.number_input("Lift mínimo", value=1.0, step=0.1)
min_ante = st.sidebar.slider("Tamanho mínimo do antecedente", 1, 4, 1)
min_cons = st.sidebar.slider("Tamanho mínimo do consequente", 1, 3, 1)
top_k_sug = st.sidebar.slider("Top-N recomendações por transação", 1, 10, 5)
produto_foco = st.sidebar.text_input("Filtrar por produto de interesse (opcional)", value="")

# --------------------------
# Processamento principal
# --------------------------
if up_tt and up_pr:
    # Leitura
    df_tt_raw = carregar_csv(up_tt)
    df_pr_raw = carregar_csv(up_pr)

    # Seleção de colunas
    st.write("#### Selecione as colunas de transações")
    c1, c2 = st.columns(2)
    with c1:
        col_tt = st.selectbox(
            "Coluna da base Treino/Teste",
            options=list(df_tt_raw.columns),
            index=list(df_tt_raw.columns).index("Transacao") if "Transacao" in df_tt_raw.columns else 0,
            key="col_tt"
        )
    with c2:
        col_pr = st.selectbox(
            "Coluna da base Predição",
            options=list(df_pr_raw.columns),
            index=list(df_pr_raw.columns).index("Transacao") if "Transacao" in df_pr_raw.columns else 0,
            key="col_pr"
        )

    # Validações de tamanho
    n_tt = len(df_tt_raw)
    n_pr = len(df_pr_raw)

    ok_tt = (n_tt >= 1000)
    ok_pr = (n_pr >= 100)

    if not ok_tt:
        st.error(f"A base de **Treino/Teste** deve ter **pelo menos 1000** transações (recebidas: {n_tt}).")
    if not ok_pr:
        st.error(f"A base de **Predição** deve ter **pelo menos 100** transações (recebidas: {n_pr}).")

    if ok_tt and ok_pr:
        # One-hot
        df_tt_bin, mlb_tt, listas_tt = preparar_ohe(df_tt_raw, col_tt)
        df_pr_bin, mlb_pr, listas_pr = preparar_ohe(df_pr_raw, col_pr)

        st.markdown("### Amostra das transações processadas")
        ca, cb = st.columns(2)
        with ca:
            st.caption("Treino/Teste — one-hot (5 primeiras)")
            st.dataframe(df_tt_bin.head())
        with cb:
            st.caption("Predição — one-hot (5 primeiras)")
            st.dataframe(df_pr_bin.head())

        # Apriori na base de treino/teste
        freq_itens = apriori(df_tt_bin, min_support=min_sup, use_colnames=True, max_len=max_len)
        if freq_itens.empty:
            st.warning("Nenhum item frequente encontrado na base de Treino/Teste. Ajuste os parâmetros.")
            st.stop()

        n_trans_tt = len(df_tt_bin)
        freq_itens["support_count"] = (freq_itens["support"] * n_trans_tt).round(0).astype(int)

        st.subheader("Itens frequentes (Treino/Teste)")
        st.dataframe(freq_itens.sort_values(["support", "itemsets"], ascending=[False, True]).reset_index(drop=True))

        # Regras
        regras = association_rules(freq_itens, metric="confidence", min_threshold=min_conf)
        if regras.empty:
            st.warning("Nenhuma regra foi gerada. Tente reduzir confiança, aumentar max_len ou reduzir suporte.")
            st.stop()

        # Filtros de regras
        regras = regras[regras["lift"] >= min_lift].copy()
        regras["ante_len"] = regras["antecedents"].apply(lambda s: len(s))
        regras["cons_len"] = regras["consequents"].apply(lambda s: len(s))
        regras = regras[(regras["ante_len"] >= min_ante) & (regras["cons_len"] >= min_cons)]

        # Conversão amigável
        regras["antecedents_txt"] = regras["antecedents"].apply(fs_to_text)
        regras["consequents_txt"] = regras["consequents"].apply(fs_to_text)

        # Filtro por produto de interesse (opcional)
        if produto_foco.strip():
            import re as _re
            p = _re.escape(produto_foco.strip().lower())
            regras = regras[
                regras["antecedents_txt"].str.contains(fr"\b{p}\b") |
                regras["consequents_txt"].str.contains(fr"\b{p}\b")
            ]

        if regras.empty:
            st.warning("Regras geradas, mas todas foram filtradas. Afrouxe os filtros ou limpe o 'produto de interesse'.")
            st.stop()

        ordem = st.selectbox("Ordenar regras por", options=["lift", "confidence", "support"], index=0)
        regras = regras.sort_values(by=ordem, ascending=False)

        cols_show = ["antecedents_txt", "consequents_txt", "support", "confidence", "lift", "leverage", "conviction"]
        st.subheader("Regras de associação (Treino/Teste)")
        st.dataframe(regras[cols_show].reset_index(drop=True), use_container_width=True)

        st.download_button(
            "⬇️ Baixar regras (CSV)",
            data=regras[cols_show].to_csv(index=False).encode("utf-8"),
            file_name="regras_apriori_treino.csv",
            mime="text/csv"
        )

        # --------------------------
        # Aplicação das Regras na Base de Predição
        # --------------------------
        st.markdown("---")
        st.subheader("Aplicação das Regras na Base de Predição")

        listas_pr_sets = [set(lst) for lst in listas_pr]
        regras_sorted = regras.sort_values(["lift", "confidence", "support"], ascending=False).copy()

        recs_por_transacao = []
        for idx, itens in enumerate(listas_pr_sets):
            recs = []
            for _, r in regras_sorted.iterrows():
                ante = r["antecedents"]
                cons = r["consequents"]
                if ante.issubset(itens):
                    sugeridos = [c for c in cons if c not in itens]
                    if sugeridos:
                        recs.append((tuple(sorted(sugeridos)), r["lift"], r["confidence"], r["support"]))
            # Ordena e aplica top-k
            recs.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
            recs = recs[:top_k_sug]
            for sug, lift, conf, sup in recs:
                recs_por_transacao.append({
                    "id_transacao": idx,
                    "itens_existentes": ", ".join(sorted(itens)),
                    "recomendacao": ", ".join(sug),
                    "lift_regra": round(lift, 4),
                    "conf_regra": round(conf, 4),
                    "support_regra": round(sup, 4)
                })

        if recs_por_transacao:
            df_recs = pd.DataFrame(recs_por_transacao)

            st.write("#### Amostra das recomendações por transação (Predição)")
            st.dataframe(df_recs.head(20), use_container_width=True)

            # Métricas de cobertura e resumo
            transacoes_com_rec = df_recs["id_transacao"].nunique()
            total_pred = len(listas_pr_sets)
            cobertura = transacoes_com_rec / total_pred if total_pred else 0.0
            media_rec_por_trans = df_recs.groupby("id_transacao").size().mean()

            st.markdown(
                f'''
**Cobertura**: {transacoes_com_rec}/{total_pred} transações ({cobertura:.0%}) receberam ao menos 1 recomendação.  
**Média de recomendações por transação (com recomendação)**: {media_rec_por_trans:.2f}
'''
            )

            # Top itens recomendados
            cont_recs = Counter()
            for r in recs_por_transacao:
                for it in [i.strip() for i in r["recomendacao"].split(",")]:
                    if it:
                        cont_recs[it] += 1
            top_itens = pd.DataFrame(cont_recs.most_common(10), columns=["item", "ocorrencias"])
            st.write("#### Top 10 itens mais recomendados")
            st.dataframe(top_itens, use_container_width=True)

            # Download das recomendações
            st.download_button(
                "⬇️ Baixar recomendações por transação (CSV)",
                data=df_recs.to_csv(index=False).encode("utf-8"),
                file_name="recomendacoes_predicao.csv",
                mime="text/csv"
            )
        else:
            st.info("Nenhuma transação da base de Predição recebeu recomendação com os filtros atuais. Afrouxe os parâmetros.")

else:
    st.info("Envie **duas bases** (Treino/Teste e Predição) ou gere as sintéticas acima para começar.")

# --------------------------
# Atividade (roteiro didático)
# --------------------------
st.markdown('''
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
   - Escolha **ao menos 2 regras** e explique como usá‑las (combos, cross‑sell, layout de gôndolas, precificação, *endcaps*).  
   - Proponha **um experimento A/B** simples para validar o impacto de uma recomendação no faturamento.

5) **Reflexões (responder no relatório)**  
   - Como **suporte** impacta a **escalabilidade operacional**?  
   - Como **confiança** e **lift** impactam a **qualidade** das recomendações?  
   - Qual trade‑off você encontrou entre **número de regras** e **relevância**?  
   - Se um item é muito popular, como isso influencia o **lift**?

> **Dica**: Foque em regras com **lift > 1** e **confiança** alta; porém, um **suporte** muito baixo pode dificultar a execução prática na loja.
''')
