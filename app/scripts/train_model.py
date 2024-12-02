# Integração do Treinamento do Modelo de Machine Learning

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

# Função para treinar o modelo RandomForest e calcular previsões
def train_model(df):
    # Preparar os dados para o modelo de Machine Learning
    df = df.sort_values('date')
    df['price_lag1'] = df['price'].shift(1)
    df.dropna(inplace=True)
    
    X = df[['price_lag1']]
    y = df['price']
    
    # Dividir os dados em treino e teste
    split_index = int(0.8 * len(df))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Treinar o modelo RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)
    
    # Avaliar a performance do modelo
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Exibir as métricas de performance
    st.write("### Métricas de Performance do Modelo")
    st.write(f"Erro Quadrático Médio (MSE): {mse:.2f}")
    st.write(f"Erro Absoluto Médio (MAE): {mae:.2f}")
    st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")
    
    # Salvar o modelo treinado
    joblib.dump(model, 'random_forest_model.pkl')
    st.success('Modelo treinado e salvo com sucesso!')
    
    return model

# Função para prever o preço com base no modelo treinado
def predict_price(model, recent_data):
    return model.predict(np.array(recent_data).reshape(1, -1))[0]

# Função para integrar o treinamento e previsão ao Streamlit
def main(df):
    st.title('Previsão do Preço do Petróleo Brent')
    st.markdown("Utilizando um modelo de Machine Learning para prever o preço do petróleo com base nos dados históricos.")
    
    if st.button('Treinar Modelo e Ver Resultados'):
        model = train_model(df)
        st.success('Modelo treinado com sucesso! Verifique as métricas de performance.')
        
        # Previsão para o próximo valor do preço com base no último valor disponível
        if st.button('Prever Próximo Preço'):
            recent_price = df['price'].iloc[-1]
            predicted_price = predict_price(model, [recent_price])
            st.write(f"### Previsão do Próximo Preço do Petróleo: ${predicted_price:.2f}")

if __name__ == '__main__':
    # Carregar os dados do dashboard
    df = pd.read_csv('preco_petroleo.csv')  # Exemplo, alterar conforme necessário
    main(df)
