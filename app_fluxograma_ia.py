# app_fluxograma_ia.py

import streamlit as st

st.set_page_config(page_title="Escolha de Modelo de IA", page_icon="🤖")

st.title("🤖 Escolha um Modelo de IA para o seu Problema de Negócio")

st.markdown("Responda às perguntas abaixo para descobrir o tipo de modelo e algoritmo ideal.")

# Primeira pergunta
objetivo = st.radio(
    "Qual é o objetivo do seu problema?",
    (
        "Prever um número (valor contínuo)",
        "Classificar algo (sim/não ou categorias)",
        "Encontrar padrões em dados não rotulados",
        "Aprender com tentativa e erro (reforço)"
    )
)

if objetivo == "Prever um número (valor contínuo)":
    st.success("✅ Recomendação: Aprendizado Supervisionado - Regressão")
    st.info("🔧 Algoritmo sugerido: **Regressão Linear**")
    
elif objetivo == "Classificar algo (sim/não ou categorias)":
    tipo_dados = st.radio(
        "Qual das situações melhor descreve seu problema?",
        (
            "Preciso prever uma resposta binária (sim/não)",
            "Meus dados são categóricos e simbólicos",
            "Quero classificar com base na proximidade entre exemplos",
            "Preciso de limites complexos entre as classes"
        )
    )
    
    st.success("✅ Recomendação: Aprendizado Supervisionado - Classificação")
    
    if tipo_dados == "Preciso prever uma resposta binária (sim/não)":
        st.info("🔧 Algoritmo sugerido: **Regressão Logística**")
    elif tipo_dados == "Meus dados são categóricos e simbólicos":
        st.info("🔧 Algoritmo sugerido: **Naïve Bayes**")
    elif tipo_dados == "Quero classificar com base na proximidade entre exemplos":
        st.info("🔧 Algoritmo sugerido: **KNN (K-Nearest Neighbors)**")
    elif tipo_dados == "Preciso de limites complexos entre as classes":
        st.info("🔧 Algoritmo sugerido: **SVM (Support Vector Machine)**")
        
elif objetivo == "Encontrar padrões em dados não rotulados":
    tipo_padrao = st.radio(
        "O que você quer descobrir?",
        (
            "Agrupar itens semelhantes",
            "Encontrar regras de associação entre itens"
        )
    )
    st.success("✅ Recomendação: Aprendizado Não-Supervisionado")
    if tipo_padrao == "Agrupar itens semelhantes":
        st.info("🔧 Algoritmo sugerido: **K-Means**")
    else:
        st.info("🔧 Algoritmo sugerido: **Apriori**")

elif objetivo == "Aprender com tentativa e erro (reforço)":
    st.success("✅ Recomendação: Aprendizado por Reforço")
    st.info("🔧 Algoritmo sugerido: **Q-Learning**")

st.markdown("---")
st.caption("Desenvolvido para apoio didático em cursos de IA aplicada aos negócios.")
