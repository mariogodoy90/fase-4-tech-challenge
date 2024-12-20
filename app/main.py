# Ajustando e Melhorando o Dashboard Interativo

import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
from scripts.train_model import train_model, predict_price
import joblib
import datetime
import os

# URL do site de onde vamos extrair os dados do preço do petróleo Brent
url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'

# Fazer a requisição para obter os dados da página
response = requests.get(url, timeout=10)
response.raise_for_status()

# Usar BeautifulSoup para fazer o parsing do HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Verificar se a tabela de dados está presente no HTML
data = []
table = soup.find('table', {'id': 'grd_DXMainTable'})  # Procurar pela tabela com o ID específico 'grd_DXMainTable'
if not table:
    st.error("Erro: Nenhuma tabela encontrada na página HTML.")
    st.stop()
else:
    # Extrair a tabela de dados
    rows = table.find_all('tr', {'class': ['dxgvDataRow', 'dxgvFocusedRow']})
    if len(rows) == 0:
        st.error("Erro: Nenhum dado encontrado na tabela.")
        st.stop()
    else:
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
            date = cols[0].text.strip()
            price = cols[1].text.strip()
            try:
                data.append({'date': date, 'price': float(price.replace('.', '').replace(',', '.'))})
            except ValueError:
                st.warning(f"Valor inválido encontrado e ignorado: {price}")

# Transformar os dados extraídos em um DataFrame
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# Configurando o Streamlit para visualização
def main():
    st.sidebar.title('Menu de Navegação')
    option = st.sidebar.radio('Selecione a visualização:', ['Página Inicial', 'Variação Histórica', 'Média Anual', 'Crises Econômicas', 'Previsões e Métricas de Performance', 'Previsão de Preços Dinâmica'])
    
    if option == 'Página Inicial':
        st.title('📊 Dashboard Interativo - Preço do Petróleo Brent - Fase 4 do Tech Challenge')
        st.markdown(
            """
            🌍 **Bem-vindo ao Dashboard Interativo do Preço do Petróleo Brent**.
            Explore a variação histórica dos preços e entenda os principais fatores que os influenciam.
            Utilize o menu ao lado para navegar por diferentes análises e insights.
            """
        )
    
    if option == 'Variação Histórica':
        # Gráfico 1: Variação Histórica do Preço
        st.subheader('📈 Variação Histórica do Preço do Petróleo')
        fig1 = px.line(df, x='date', y='price', title='Preço do Petróleo Brent ao Longo do Tempo', line_shape='linear', labels={'date': 'Ano', 'price': 'Preço (US$)'}, color_discrete_sequence=['#1f77b4'])
        fig1.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16, margin=dict(l=40, r=40, t=60, b=40), legend_title_text='Legenda')
        fig1.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True, title_font=dict(size=14), tickfont=dict(size=12))
        fig1.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))
        fig1.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True)
        fig1.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True)

        # Adicionar marcadores para as crises econômicas
        crises = [
            {'name': 'Crise Financeira de 2008', 'date': '2008-09-01'},
            {'name': 'Queda dos Preços do Petróleo em 2014', 'date': '2014-06-01'},
            {'name': 'Pandemia COVID-19', 'date': '2020-03-01'}
        ]

        for i, crise in enumerate(crises):
            data_crise = pd.to_datetime(crise['date'])
            ano = data_crise.year
            mes = data_crise.month
            
            # Filtrar por ano e mês para garantir a precisão do posicionamento
            df_crise = df[(df['date'].dt.year == ano) & (df['date'].dt.month == mes)]
            if not df_crise.empty:
                y_value = df_crise['price'].values[0]
                fig1.add_annotation(
                    x=data_crise,  # Usar a data completa para uma anotação precisa
                    y=y_value,
                    text=crise['name'],
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40 if i % 2 == 0 else 40,  # Alternar entre anotações acima e abaixo da linha do gráfico
                    bgcolor='rgba(0, 0, 0, 0.8)',  # Fundo preto para melhor contraste
                    bordercolor='white',
                    font=dict(size=12, color='white'),
                    arrowcolor='white'
                )

        st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
        st.markdown("💡 **Insight 1**: O preço do petróleo apresenta picos significativos durante crises geopolíticas, como guerras ou tensões no Oriente Médio.")
    
    elif option == 'Média Anual':
        # Gráfico 2: Preço Médio por Ano
        if not df.empty:
            df['year'] = df['date'].dt.year
            yearly_avg = df.groupby('year')['price'].mean().reset_index()
            st.subheader('📊 Preço Médio Anual do Petróleo')
            fig2 = px.bar(yearly_avg, x='year', y='price', title='Preço Médio Anual do Petróleo Brent', labels={'year': 'Ano', 'price': 'Preço (US$)'}, color_discrete_sequence=['#ff7f0e'])
            fig2.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True)
            fig2.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True)
            st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
            st.markdown("💡 **Insight 2**: A média anual do preço do petróleo reflete claramente os impactos de eventos econômicos globais, como recessões e crises financeiras.")
    
    elif option == 'Crises Econômicas':
        # Gráfico 3: Análise de Crises Econômicas
        st.subheader('📉 Impacto de Crises Econômicas no Preço do Petróleo')

        crises = [
            {'name': 'Crise Financeira de 2008', 'start': '2008-09-01', 'end': '2009-06-30'},
            {'name': 'Pandemia COVID-19', 'start': '2020-03-01', 'end': '2020-12-31'},
            {'name': 'Queda dos Preços do Petróleo em 2014', 'start': '2014-06-01', 'end': '2016-02-29'},
            {'name': 'Guerra na Ucrânia', 'start': '2022-02-01', 'end': '2022-12-31'}
        ]

        crise_selecionada = st.selectbox('Selecione a crise econômica que deseja visualizar:', [crise['name'] for crise in crises])
        crise = next(item for item in crises if item['name'] == crise_selecionada)
        mask = (df['date'] >= crise['start']) & (df['date'] <= crise['end'])
        if not df[mask].empty:
            fig3 = px.line(df[mask], x='date', y='price', title=f'Impacto da {crise['name']} no Preço do Petróleo', labels={'date': 'Data', 'price': 'Preço'}, line_shape='linear', color_discrete_sequence=['#2ca02c'])
            fig3.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16, margin=dict(l=40, r=40, t=60, b=40), legend_title_text='Legenda')
            fig3.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True, title_font=dict(size=14), tickfont=dict(size=12))
            fig3.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))
            fig3.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True)
            fig3.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True)
            st.plotly_chart(fig3, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
            st.markdown(f"Insight: {crise['name']} teve um impacto significativo no preço, refletindo incertezas econômicas globais.")

    elif option == 'Previsões e Métricas de Performance':
        # Chamando a função de treinamento do modelo de Machine Learning
        st.subheader('🔮 Previsões e Métricas de Performance')
        st.markdown(
            """
            Utilizando um modelo de Machine Learning para prever o preço do petróleo com base nos dados históricos. 
            O modelo é treinado com dados históricos do preço do petróleo e utiliza um algoritmo de **Random Forest Regressor** para fazer previsões futuras. 
            O gráfico abaixo compara os preços reais do petróleo com os valores previstos pelo modelo, permitindo uma avaliação visual da precisão do modelo.
            As métricas de performance, como **Erro Quadrático Médio (MSE)**, **Erro Absoluto Médio (MAE)** e **Coeficiente de Determinação (R²)**, são apresentadas para avaliar quantitativamente a qualidade do modelo.
            """
        )
        
        # Usando session_state para armazenar o modelo treinado
        if 'model' not in st.session_state:
            st.session_state['model'] = None

        if os.path.exists('random_forest_model.pkl') and st.session_state['model'] is None:
            st.session_state['model'] = joblib.load('random_forest_model.pkl')
            st.success('Modelo carregado com sucesso!')
        
        if st.session_state['model'] is None:
            if st.button('Treinar Modelo e Ver Resultados', key='treinar_modelo'):
                st.session_state['model'] = train_model(df)
                joblib.dump(st.session_state['model'], 'random_forest_model.pkl')
                st.success('Modelo treinado e salvo com sucesso! Verifique as métricas de performance.')
                
        if st.session_state['model'] is not None:
            # Fazer previsões e gerar gráfico comparativo
            X = df['price'].shift(1).dropna().values.reshape(-1, 1)
            y_real = df['price'][1:].values
            y_pred = st.session_state['model'].predict(X)

            st.subheader('Métricas de Performance do Modelo')
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_real, y_pred)
            mae = mean_absolute_error(y_real, y_pred)
            r2 = r2_score(y_real, y_pred)

            # Exibindo as métricas em uma tabela estruturada
            metrics_df = pd.DataFrame({
                'Métrica': ['Erro Quadrático Médio (MSE)', 'Erro Absoluto Médio (MAE)', 'Coeficiente de Determinação (R²)'],
                'Valor': [f'{mse:.2f}', f'{mae:.2f}', f'{r2:.2f}']
            })
            st.table(metrics_df)

            comparativo_df = pd.DataFrame({'Data': df['date'][1:], 'Preço Real': y_real, 'Preço Previsto': y_pred})
            fig4 = px.line(comparativo_df, x='Data', y=['Preço Real', 'Preço Previsto'], title='Comparação entre Preço Real e Preço Previsto - Atualizado', labels={'Data': 'Ano', 'value': 'Preço (US$)'}, line_shape='linear', color_discrete_sequence=['#d62728', '#9467bd'])
            fig4.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16, margin=dict(l=40, r=40, t=60, b=40), legend_title_text='Legenda')
            fig4.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True, title_font=dict(size=14), tickfont=dict(size=12))
            fig4.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))
            fig4.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True)
            fig4.update_xaxes(tickformat='%Y', fixedrange=True, showgrid=True)
            st.plotly_chart(fig4, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
            st.markdown("💡 **Insight**: O gráfico compara os preços reais do petróleo com os valores previstos pelo modelo, permitindo avaliar a precisão do modelo.")

    elif option == 'Previsão de Preços Dinâmica':
        # Verificar se o modelo foi treinado antes de prosseguir
        if st.session_state.get('model') is None:
            st.warning('⚠️ O modelo de Machine Learning precisa ser treinado antes de fazer previsões. Vá para a seção "Previsão de Preços e Métricas de Performance" e clique em "Treinar Modelo".')
            st.stop()

        st.subheader('🌟 Previsão de Preço Dinâmica para os Próximos Dias 🌟')
        st.markdown('Use o slider abaixo para selecionar o número de dias que deseja prever o preço do petróleo. A previsão será exibida em forma de gráfico e tabela para facilitar a visualização.')

        # Previsão dinâmica de preço para os próximos dias
        dias_para_prever = st.slider('Selecione o número de dias para prever', min_value=1, max_value=30, value=1)

        if st.button('Prever Preço', key='prever_preco_dinamico'):
            ultimo_preco = df[df['date'] == df['date'].max()]['price'].values[-1]
            previsoes = []
            data_atual = datetime.datetime.now()
            
            for i in range(1, dias_para_prever + 1):
                proxima_data = data_atual + datetime.timedelta(days=i)
                previsao = st.session_state['model'].predict([[ultimo_preco]])[0]
                previsoes.append((proxima_data.strftime('%Y-%m-%d'), previsao))
                ultimo_preco = previsao  # Atualizar o último preço com a previsão atual
            
            if dias_para_prever == 1:
                # Exibir a previsão apenas em texto, sem gerar o gráfico
                st.write(f"Previsão do preço do petróleo para {previsoes[0][0]}: ${previsoes[0][1]:.2f}")
            else:
                # Criar um DataFrame para as previsões
                previsoes_df = pd.DataFrame(previsoes, columns=['Data', 'Preço Previsto (US$)'])
                previsoes_df['Data'] = pd.to_datetime(previsoes_df['Data'])

                # Limpar o espaço antes de renderizar um novo gráfico
                grafico_espaco = st.empty()

                # Criar um gráfico de linha para as previsões
                fig5 = px.line(previsoes_df, x='Data', y='Preço Previsto (US$)', title='Previsão do Preço do Petróleo para os Próximos Dias')
                fig5.update_layout(
                    title_font_size=20,
                    xaxis_title_font_size=16,
                    yaxis_title_font_size=16,
                    margin=dict(l=40, r=40, t=60, b=40),
                    xaxis=dict(
                        tickformat='%b %d',
                        tickangle=45,  # Inclinar os rótulos para melhor visualização
                        nticks=min(len(previsoes), 30)  # Ajustar o número de marcas no eixo x para se adequar ao número de previsões (máx. 30)
                    )
                )

                # Exibir o gráfico no espaço reservado
                grafico_espaco.plotly_chart(fig5, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

                # Exibir a tabela de previsões
                st.write('### Tabela de Previsões do Preço do Petróleo')
                st.table(previsoes_df)

if __name__ == '__main__':
    main()
