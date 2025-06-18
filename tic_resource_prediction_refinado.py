
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------
# Simular dados fictícios com relação realista
# ------------------------------
def gerar_dados_ficticios(n=500):
    np.random.seed(42)
    tipo_recurso = np.random.choice(['Notebook', 'Desktop', 'Nobreak', 'Impressora'], size=n)
    tempo_uso_meses = np.random.randint(1, 60, size=n)
    num_manutencoes = np.random.randint(0, 6, size=n)  # 0 a 5
    num_colaboradores_unidade = np.random.randint(1, 31, size=n)  # 1 a 30

    # Gerar o tempo até substituição com lógica
    # Base: 72 meses, menos penalizações
    tempo_ate_substituicao = (
        72
        - (num_manutencoes * 6)  # cada manutenção reduz 6 meses
        - (num_colaboradores_unidade * 2)  # cada colaborador reduz 2 meses
        - (tempo_uso_meses * 0.5)  # o tempo de uso também impacta
    )
    tempo_ate_substituicao = np.clip(tempo_ate_substituicao, 6, 72)

    data = pd.DataFrame({
        'tipo_recurso': tipo_recurso,
        'tempo_uso_meses': tempo_uso_meses,
        'num_manutencoes': num_manutencoes,
        'num_colaboradores_unidade': num_colaboradores_unidade,
        'tempo_ate_substituicao': tempo_ate_substituicao
    })

    return data

# ------------------------------
# Treinar modelo
# ------------------------------
def treinar_modelo(data):
    data_encoded = pd.get_dummies(data, columns=['tipo_recurso'])
    
    X = data_encoded.drop(columns=['tempo_ate_substituicao'])
    y = data_encoded['tempo_ate_substituicao']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    
    return model, X.columns, mae, rmse

# ------------------------------
# App Streamlit
# ------------------------------
def main():
    st.title("Previsão de Substituição de Recursos de TIC (Refinado)")
    
    data = gerar_dados_ficticios()
    model, features, mae, rmse = treinar_modelo(data)
    
    st.write("Modelo treinado em dados fictícios realistas.")
    st.write(f"Erro médio absoluto (MAE): {mae:.2f} meses")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} meses")
    
    st.header("Insira os dados do recurso para previsão")
    
    tipo_recurso = st.selectbox("Tipo de recurso", ['Notebook', 'Desktop', 'Nobreak', 'Impressora'])
    tempo_uso_meses = st.slider("Tempo de uso (meses)", 1, 60, 12)
    num_manutencoes = st.slider("Número de manutenções", 0, 5, 0)
    num_colaboradores_unidade = st.slider("Número de colaboradores usando o recurso", 1, 30, 5)
    
    input_dict = {
        'tempo_uso_meses': tempo_uso_meses,
        'num_manutencoes': num_manutencoes,
        'num_colaboradores_unidade': num_colaboradores_unidade,
        'tipo_recurso_Desktop': 1 if tipo_recurso == 'Desktop' else 0,
        'tipo_recurso_Impressora': 1 if tipo_recurso == 'Impressora' else 0,
        'tipo_recurso_Nobreak': 1 if tipo_recurso == 'Nobreak' else 0,
        'tipo_recurso_Notebook': 1 if tipo_recurso == 'Notebook' else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]
    
    if st.button("Prever tempo até substituição"):
        previsao = model.predict(input_df)[0]
        st.success(f"Tempo estimado até substituição: {previsao:.1f} meses")

if __name__ == "__main__":
    main()
