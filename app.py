{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 AppleColorEmoji;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
from sklearn.feature_extraction.text import CountVectorizer\
from sklearn.naive_bayes import MultinomialNB\
from sklearn.pipeline import Pipeline\
\
# Dados de treino (exemplo)\
emails = [\
    "Oferta especial! Compre agora",\
    "Promo\'e7\'e3o exclusiva para voc\'ea",\
    "\'daltima chance",\
    "Documento solicitado",\
    "Reuni\'e3o amanh\'e3 \'e0s 10h",\
    "Relat\'f3rio financeiro do trimestre"\
]\
labels = [1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = n\'e3o spam\
\
# Treinando o modelo com pipeline\
modelo = Pipeline([\
    ('vectorizer', CountVectorizer()),\
    ('classifier', MultinomialNB())\
])\
modelo.fit(emails, labels)\
\
# --- Interface com Streamlit ---\
st.title("
\f1 \uc0\u55357 \u56589 
\f0  Classificador de E-mails Spam")\
st.write("Digite o conte\'fado de um e-mail abaixo:")\
\
entrada = st.text_area("Texto do e-mail")\
\
if st.button("Classificar"):\
    if entrada.strip() == "":\
        st.warning("Por favor, digite um e-mail para classificar.")\
    else:\
        resultado = modelo.predict([entrada])[0]\
        prob = modelo.predict_proba([entrada])[0][1]\
\
        if resultado == 1:\
            st.error(f"
\f1 \uc0\u55357 \u57000 
\f0  Este e-mail foi classificado como **SPAM** (confian\'e7a: \{prob:.2%\})")\
        else:\
            st.success(f"
\f1 \uc0\u9989 
\f0  Este e-mail **n\'e3o \'e9 SPAM** (confian\'e7a: \{prob:.2%\})")}