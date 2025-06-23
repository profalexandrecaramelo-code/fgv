
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.title("🔍 Classificador de E-mails Spam com Naive Bayes")

st.markdown("""
Este app utiliza **Aprendizado Supervisionado** com o algoritmo **Naive Bayes Multinomial**, amplamente usado em classificação de textos (como spam x não spam).

📌 **Como funciona?**
- O **CountVectorizer** transforma o texto em uma matriz de contagem de palavras (Bag of Words).
- O **Naive Bayes Multinomial** calcula a probabilidade do e-mail ser spam ou não, assumindo independência entre as palavras (modelo "ingênuo").
- O modelo aprende com um conjunto de e-mails já classificados e usa esse aprendizado para prever novos casos.

💡 **Aplicação no negócio:**  
Ajuda a automatizar o filtro de e-mails, reduzir fraudes, melhorar o atendimento ao cliente e proteger sistemas.
""")

# Dados mais ricos
emails = [
    # Spam
    "Oferta especial! Compre agora mesmo e ganhe bônus",
    "Promoção exclusiva: clique e aproveite",
    "Última chance para ganhar um prêmio incrível",
    "Grande oportunidade: crédito aprovado sem consulta",
    "Oferta limitada: baixe grátis",
    "Você foi sorteado! Seu prêmio está aqui",
    "Ganhe dinheiro rápido sem sair de casa",
    "Clique para desbloquear sua oferta secreta",
    "Ganhe viagens grátis para o Caribe",
    "Seu cartão de crédito foi pré-aprovado",
    # Não spam
    "Reunião agendada para amanhã às 10h",
    "Relatório financeiro do trimestre disponível",
    "Prezado colaborador, seguem as orientações",
    "Documentos anexos para sua revisão",
    "Atualização do projeto enviada",
    "Confirmação de inscrição no evento",
    "Parabéns pelo desempenho no último mês",
    "Horário da entrevista confirmado",
    "Seu extrato bancário está disponível",
    "Pedido de orçamento recebido"
]
labels = [1]*10 + [0]*10

# Treinamento
modelo = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
modelo.fit(emails, labels)

# Entrada do usuário
st.write("Digite o conteúdo de um e-mail abaixo para classificar:")
entrada = st.text_area("Texto do e-mail")

if st.button("Classificar"):
    if entrada.strip() == "":
        st.warning("Por favor, digite um e-mail para classificar.")
    else:
        resultado = modelo.predict([entrada])[0]
        prob = modelo.predict_proba([entrada])[0][1]

        if resultado == 1:
            st.error(f"🚨 Este e-mail foi classificado como **SPAM** (confiança: {prob:.2%})")
        else:
            st.success(f"✅ Este e-mail **não é SPAM** (confiança: {prob:.2%})")
