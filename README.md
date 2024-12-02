# Dashboard Interativo - Preço do Petróleo Brent

## Fase 4 do Tech Challenge

Este projeto consiste em um dashboard interativo desenvolvido com Streamlit para explorar dados históricos do preço do petróleo Brent. O objetivo é fornecer insights visuais sobre a evolução dos preços e os fatores que os influenciam, utilizando uma interface fácil de usar.

**URL**: https://fase-4-tech-challenge.streamlit.app/

### Funcionalidades

1. **Página Inicial**: Visão geral do dashboard e instruções de uso.
2. **Variação Histórica**: Visualização do preço do petróleo ao longo do tempo em um gráfico de linhas interativo.
3. **Média Anual**: Gráfico de barras que exibe a média anual dos preços do petróleo, permitindo uma análise mais simplificada dos dados.
4. **Crises Econômicas**: Análise do impacto de eventos econômicos significativos nos preços do petróleo, como a crise financeira de 2008, a pandemia de COVID-19, e outros.
5. **Previsões e Métricas de Performance**: Utilização de um modelo de Machine Learning (Random Forest Regressor) para prever preços futuros e exibir métricas de desempenho do modelo.
6. **Previsão de Preços Dinâmica**: Ferramenta que permite prever o preço do petróleo para até 30 dias no futuro, utilizando um slider interativo. Esta funcionalidade destaca-se pela flexibilidade e a interatividade que oferece ao usuário.


### Como Executar

1. **Instalar Dependências**:
   - Certifique-se de que possui o Python 3.x instalado.
   - Instale as dependências utilizando o arquivo `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```

2. **Executar o Dashboard**:
   - Para iniciar o dashboard, execute o seguinte comando no terminal:
     ```
     streamlit run nome_do_script.py
     ```

### Dependências

- **Python 3.x**
- **Pandas**: Manipulação e análise de dados.
- **Plotly**: Criação de gráficos interativos.
- **Streamlit**: Desenvolvimento da interface do dashboard.
- **Requests** e **BeautifulSoup**: Para fazer scraping dos dados históricos do preço do petróleo.
- **Scikit-Learn**: Para o treinamento do modelo de previsão de preços.
- **Joblib**: Para salvar e carregar o modelo treinado.

### Observações

- **Conexão com a Internet**: É necessário que a conexão esteja ativa para que os dados do preço do petróleo sejam extraídos corretamente do site fonte.
- **Treinamento do Modelo**: Caso o modelo ainda não esteja treinado, utilize o botão 'Treinar Modelo' na seção de Previsão de Preços para treinar o modelo antes de gerar previsões.

### Estrutura do Projeto

- **scripts/**: Pasta contendo scripts auxiliares, como o script para treinar e prever os preços.
- **app/**: Contém o script principal (`main.py`) que executa o dashboard.
- **random_forest_model.pkl**: Arquivo contendo o modelo treinado (se gerado).

### Plano de Deploy

#### Deploy no Streamlit Cloud

Para disponibilizar o dashboard no Streamlit Cloud, siga os seguintes passos detalhados:

1. **Preparação do Repositório**:
   - Certifique-se de que todo o código do projeto está em um repositório Git, como GitHub, GitLab ou Bitbucket.
   - Inclua um arquivo `requirements.txt` com todas as dependências do projeto para garantir que o ambiente do Streamlit possa ser configurado corretamente.

2. **Acessar o Streamlit Cloud**:
   - Acesse o [Streamlit Cloud](https://streamlit.io/cloud).
   - Faça login utilizando sua conta do GitHub, Google, ou outra opção disponível.

3. **Configuração do Projeto**:
   - Clique em **'New app'** para criar uma nova aplicação.
   - Escolha o repositório Git onde seu projeto está armazenado.
   - Selecione o branch correto (geralmente `main` ou `master`) e, no campo de **file path**, forneça o caminho para o script principal do seu dashboard, por exemplo: `app/main.py`.

4. **Configurações Avançadas** (opcional):
   - Você pode definir variáveis de ambiente necessárias para o projeto.
   - Pode optar por manter o deploy público ou privado, dependendo de suas necessidades.

5. **Executar o Deploy**:
   - Clique em **'Deploy'** para iniciar o processo.
   - O Streamlit Cloud irá criar um ambiente virtual, instalar as dependências e iniciar seu aplicativo automaticamente.

6. **Gerenciamento do Deploy**:
   - Após o deploy inicial, você pode gerenciar a aplicação através do painel do Streamlit Cloud.
   - Qualquer modificação no código fonte (no GitHub) pode disparar uma atualização automática do app.
   - Você pode pausar ou deletar a aplicação quando necessário.

7. **URL do Dashboard**:
   - Após a conclusão do deploy, o Streamlit fornecerá uma URL pública que você pode compartilhar com outros para acesso ao dashboard.