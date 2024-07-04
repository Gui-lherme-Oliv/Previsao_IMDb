# Projeto para criação de modelo para previsão de nota do IMDb a partir de dados cinematográficos

## Pré-requisitos
- Python 3.6 ou superior
- pip (gerenciador de pacotes do Python)

## Instalação
1. Clone o repositório:
    ```sh
    git clone https://github.com/Gui-lherme-Oliv/Previsao_IMDb
    cd Previsao_IMDb/
    ```

2. Crie um ambiente virtual:
    ```sh
    python -m venv venv
    ```

3. Ative o ambiente virtual:

    - No Windows:
        ```sh
        venv\Scripts\activate
        ```

    - No macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Instale as dependências:
    ```sh
    pip install -r requisitos.txt
    ```

## Execução do Projeto
1. Coloque o arquivo CSV (`desafio_indicium_imdb.csv`) na pasta do projeto.

2. Execute o notebook `modelo.ipynb` para realizar a análise e o treinamento do modelo. Pode ser utilizado o Jupyter Notebook ou Jupyter Lab para isso:
    ```sh
    jupyter notebook modelo.ipynb
    ```

## Exemplo de uso
O notebook `modelo.ipynb` contém exemplos detalhados de como carregar os dados, realizar o pré-processamento, treinar o modelo e fazer previsões. Aqui está um exemplo de como prever a nota IMDb de um novo filme:

```python
#Dados de exemplo para um novo filme
dados_filme = {
    'Released_Year': [2021],
    'Runtime': ['150 min'],
    'Meta_score': [85],
    'No_of_Votes': [2343110],
    'Gross': ['28,341,469']
}

df_filme = pd.DataFrame(dados_filme)

#Aplicando o mesmo pré-processamento aos dados
df_filme = df_filme.drop(columns=['Overview', 'Series_Title'])
df_filme['Released_Year'] = pd.to_numeric(df_filme['Released_Year'], errors='coerce')
df_filme['Runtime'] = df_filme['Runtime'].str.replace(' min', '').astype(float)
df_filme['Gross'] = df_filme['Gross'].str.replace(',', '').astype(float)
df_filme['Meta_score'] = pd.to_numeric(df_filme['Meta_score'], errors='coerce')
df_filme['No_of_Votes'] = pd.to_numeric(df_filme['No_of_Votes'], errors='coerce')

#Carregando o modelo treinado
import joblib
pipeline_completo = joblib.load('modeloRFR.pkl')

#Realizando previsões
avaliacao_pred = pipeline_completo.predict(df_filme)
print(f'Nota IMDb prevista: {round(avaliacao_pred[0], 1)}')
