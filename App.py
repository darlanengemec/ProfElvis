import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np


# Função para carregar o dicionário de modelos
def load_models():
    with open('trained_models_MAX_Q_FACTOR.pkl', 'rb') as file:
        model_dict = pickle.load(file)
    return model_dict


# Função para fazer a previsão
def fazer_previsao(Att, Disp_i, Disp_f, p, model, model_type):
    n_samples = int((Disp_f - Disp_i) / p) + 1

    x1 = []
    for i in range(n_samples):
        x1.append([Att])

    x2 = []
    for i in range(n_samples):
        valor = Disp_i + i * p
        x2.append([valor])

    x = np.concatenate((x1, x2), axis=1)
    x = pd.DataFrame(x, columns=['Att', 'Disp'])

    # Fazer a previsão com o modelo carregado
    prediction = model.predict(x)

    # Plotar o gráfico com tamanho reduzido (6x4)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x['Disp'], prediction, marker="o", label=f'Modelo: {model_type}')

    # Adicionar o valor da Atenuação à linha do gráfico
    ax.plot(Disp_i, model.predict([[Att, Disp_i]]), marker="o", markersize=8, color='red', label=f'Atenuação: {Att}')

    ax.set_title('Regressor')
    ax.set_xlabel('Dispersão')
    ax.set_ylabel('MAX_Q_FACTOR')

    # Diminuir o tamanho da fonte no eixo x
    ax.tick_params(axis='x', labelrotation=45)  # Rotacionar rótulos do eixo x em 45 graus

    # Adicionar marcações no eixo da Dispersão
    ax.set_xticks(np.arange(Disp_i, Disp_f + 1, p))
    ax.legend()

    return fig, x, prediction


# Configurações do Streamlit
st.set_page_config(page_title="Ensino de Comunicações Ópticas Potencializado por Machine Learning", layout="wide")

# Exibir as imagens
col1, col2, col3 = st.columns(3)  # Crie 3 colunas para as imagens

# Redimensione e exiba a primeira imagem
with col1:
    logo_fotonica = Image.open("assets/images/logo_fotonica.jpeg")
    logo_fotonica = logo_fotonica.resize((200, 200))  # Redimensione para (50, 50)
    st.image(logo_fotonica, use_column_width=False, width=200)

# Redimensione e exiba a segunda imagem
with col2:
    logo_ifce = Image.open("assets/images/logo_ifce.jpeg")
    logo_ifce = logo_ifce.resize((100, 100))  # Redimensione para (50, 50)
    st.image(logo_ifce, use_column_width=False, width=200)

# Redimensione e exiba a terceira imagem
with col3:
    logo_renoen = Image.open("assets/images/logo_Renoen.jpeg")
    logo_renoen = logo_renoen.resize((200, 200))  # Redimensione para (50, 50)
    st.image(logo_renoen, use_column_width=False, width=200)

# Inserir uma linha horizontal
st.markdown("<hr/>", unsafe_allow_html=True)

st.title("Ensino de Comunicações Ópticas Potencializado por Machine Learning")

# Carregar o dicionário de modelos
model_dict = load_models()

# Mapear os nomes completos dos modelos para as siglas corretas
modelo_siglas = {
    'Decision Tree': 'dtr',
    'Random Forest': 'rfr',
    'Gradient Boosting': 'gbr'
}

# Criar uma caixa de seleção para escolher o tipo de modelo
model_type = st.selectbox('Selecione o tipo de modelo', list(modelo_siglas.keys()))

# Criar uma caixa de seleção para escolher a proporção de dados
proporcao = st.slider('Selecione a proporção de treinamento (%)', 10, 90, 10, 10)

# Obter a sigla do modelo com base no nome completo
modelo_sigla = modelo_siglas[model_type]

# Criar o ID do modelo com base nas seleções do operador
model_id = f"{modelo_sigla}_{proporcao}"

# Verificar se o modelo escolhido existe no dicionário
if model_id in model_dict:
    model = model_dict[model_id]
    st.write(f'Modelo selecionado: {model_type} treinado com {proporcao}% de dados de treinamento.')

# Solicitar entrada do usuário
Att = st.number_input("Digite o valor para Atenuação: ", key="Att")
if Att and isinstance(Att, (int, float)):
    Disp_i = st.number_input("Digite o valor inicial para Dispersão: ", key="Disp_i")
    if Disp_i and isinstance(Disp_i, (int, float)):
        Disp_f = st.number_input("Digite o valor final para Dispersão: ", key="Disp_f")
        if Disp_f and isinstance(Disp_f, (int, float)):
            p = st.number_input("Digite o valor do incremento para Dispersão: ", key="p")
            if p and isinstance(p, (int, float)):
                # Chamar a função de fazer previsão aqui
                if st.button("Calcular Previsão"):
                    fig, input_values, predictions = fazer_previsao(Att, Disp_i, Disp_f, p, model, model_type)

                    # Criar colunas para exibir gráfico e tabela na mesma linha
                    col1, col2 = st.columns(2)

                    # Mostrar o gráfico na primeira coluna
                    with col1:
                        st.pyplot(fig)

                    # Mostrar a tabela na segunda coluna
                    with col2:
                        st.subheader('Valores de Entrada e Resultados do Modelo')
                        st.table(pd.concat([input_values, pd.DataFrame({'MAX_Q_FACTOR': predictions})], axis=1))
