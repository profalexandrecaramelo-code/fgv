import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import numpy as np

st.title("Sistema de Recomendação Inteligente de Produtos")

# Upload da base
uploaded_file = st.file_uploader("Faça o upload da base de clientes (.csv)", type=["csv"])

if uploaded_file:
    dados = pd.read_csv(uploaded_file)
    st.subheader("Base de Dados Carregada")
    st.dataframe(dados.head())

    # Selecionar cliente fictício
    cliente_selecionado = st.selectbox("Selecione um cliente para gerar recomendações:", dados['cliente_id'])

    # Definir colunas de entrada
    features = ['idade', 'num_compras', 'valor_medio_compra', 'avaliacao_media', 
                'tempo_medio_sessao_min', 'num_itens_visualizados']

    X = dados[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Separar dados em treino e cliente selecionado
    cliente_index = dados[dados['cliente_id'] == cliente_selecionado].index[0]
    cliente_dados = X_scaled[cliente_index].reshape(1, -1)

    X_treino = np.delete(X_scaled, cliente_index, axis=0)
    ids_treino = dados.drop(index=cliente_index)['cliente_id'].values

    # Treinar modelo
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X_treino)

    # Fazer recomendações
    distancias, indices = knn.kneighbors(cliente_dados)
    recomendados = ids_treino[indices[0]]

    st.subheader("Clientes com perfil semelhante:")
    st.write(recomendados)

    # Simular recomendação de produtos com base na categoria mais comprada
    categoria_cliente = dados.loc[cliente_index, 'categoria_mais_comprada']
    st.subheader("Produtos recomendados para este perfil")
    st.markdown(f"**Categoria preferencial identificada:** {categoria_cliente}")

    produtos_mock = {
        'Tecnologia': ['Notebook Lenovo', 'Smartphone Samsung', 'Fone de ouvido JBL'],
        'Casa e Jardim': ['Furadeira Bosch', 'Kit Jardinagem', 'Luminária LED'],
        'Moda': ['Camisa Polo', 'Tênis Adidas', 'Bolsa Feminina'],
        'Esporte': ['Bicicleta', 'Luvas de Boxe', 'Corda de Pular'],
        'Livros': ['A Revolução dos Bichos', 'O Poder do Hábito', 'Do Mil ao Milhão'],
        'Beleza': ['Perfume Importado', 'Base Maybelline', 'Creme Anti-idade'],
        'Pet': ['Ração Golden', 'Brinquedo Interativo', 'Shampoo Pet Clean'],
        'Brinquedos': ['Lego Star Wars', 'Boneca Barbie', 'Jogo Educativo']
    }

    produtos = produtos_mock.get(categoria_cliente, ['Produto 1', 'Produto 2'])
    for prod in produtos:
        st.markdown(f"- {prod}")

    # Métricas simuladas de recomendação
    st.subheader("Métricas de Avaliação da Recomendação (simuladas)")
    st.metric("Precisão", "82.5%")
    st.metric("Cobertura", "75.0%")
    st.metric("Diversidade", "68.0%")
