import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import tempfile
import os
import xlsxwriter
import base64

# Função para carregar o dicionário de modelos
def load_models():
    with open('trained_models_MAX_Q_FACTOR.pkl', 'rb') as file1, open('trained_models_MIN_BER.pkl', 'rb') as file2:
        model_dict_max_q = pickle.load(file1)
        model_dict_min_ber = pickle.load(file2)
    return model_dict_max_q, model_dict_min_ber

# Função para fazer a previsão de MAX_Q_FACTOR
def fazer_previsao_max_q(Att, Disp_i, Disp_f, p, model, model_type):
    # Código para calcular MAX_Q_FACTOR
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

    prediction = model.predict(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x['Disp'], prediction, marker="o", label=f'Modelo: {model_type}')

    ax.plot(Disp_i, model.predict([[Att, Disp_i]]), marker="o", markersize=8, color='red', label=f'Atenuação: {Att}')

    ax.set_title('Gráfico da Predição')
    ax.set_xlabel('Dispersão [ps/nm/km]')
    ax.set_ylabel('MAX_Q_FACTOR')

    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xticks(np.arange(Disp_i, Disp_f + 1, p))
    ax.legend()

    return fig, x, prediction

# Função para fazer a previsão de MIN_BER
def fazer_previsao_min_ber(Att, Disp_i, Disp_f, p, model, model_type):
    # Código para calcular MIN_BER
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

    prediction = model.predict(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x['Disp'], prediction, marker="o", label=f'Modelo: {model_type}')

    ax.plot(Disp_i, model.predict([[Att, Disp_i]]), marker="o", markersize=8, color='red', label=f'Atenuação: {Att}')

    ax.set_title('Gráfico da Predição')
    ax.set_xlabel('Dispersão [ps/nm/km]')
    ax.set_ylabel('MIN_BER')

    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xticks(np.arange(Disp_i, Disp_f + 1, p))
    ax.legend()

    return fig, x, prediction

# Configurações do Streamlit
st.set_page_config(page_title="Ensino de Comunicações Ópticas Potencializado por Machine Learning", layout="wide")

# Exibir as imagens
col1, col2, col3 = st.columns(3)

with col1:
    logo_fotonica = Image.open("assets/images/logo_fotonica.jpeg")
    logo_fotonica = logo_fotonica.resize((200, 200))
    st.image(logo_fotonica, use_column_width=False, width=200)

with col2:
    logo_ifce = Image.open("assets/images/logo_ifce.jpeg")
    logo_ifce = logo_ifce.resize((100, 100))
    st.image(logo_ifce, use_column_width=False, width=200)

with col3:
    logo_renoen = Image.open("assets/images/logo_Renoen.jpeg")
    logo_renoen = logo_renoen.resize((200, 200))
    st.image(logo_renoen, use_column_width=False, width=200)

st.markdown("<hr/>", unsafe_allow_html=True)

# Alterando o tamanho da fonte do título
st.markdown('<h1 style="font-size:33px;">Ensino de Comunicações Ópticas Potencializado por Machine Learning</h1>', unsafe_allow_html=True)

# Carregar o dicionário de modelos
model_dict_max_q, model_dict_min_ber = load_models()

modelo_siglas = {
    'Decision Tree': 'dtr',
    'Random Forest': 'rfr',
    'Gradient Boosting': 'gbr'
}

model_type = st.selectbox('Selecione o tipo de modelo', list(modelo_siglas.keys()))
proporcao = st.slider('Selecione a proporção de treinamento (%)', 10, 90, 10, 10)

modelo_sigla = modelo_siglas[model_type]
model_id = f"{modelo_sigla}_{proporcao}"

if model_id in model_dict_max_q and model_id in model_dict_min_ber:
    model_max_q = model_dict_max_q[model_id]
    model_min_ber = model_dict_min_ber[model_id]
    st.write(f'Modelo selecionado: {model_type} treinado com {proporcao}% de dados de treinamento (MAX_Q_FACTOR e MIN_BER).')

fig_max_q = None
fig_min_ber = None
input_values_max_q = None
input_values_min_ber = None
predictions_max_q = None
predictions_min_ber = None

# Seção de entrada de dados
Att = st.number_input("Digite o valor para Atenuação [dB/km] (Digite valores entre 0,1 e 0,6): ", key="Att")
if Att and isinstance(Att, (int, float)):
    Disp_i = st.number_input("Digite o valor inicial para Dispersão [ps/nm/km] (Digite valores entre 12,75 e 17,75): ", key="Disp_i")
    if Disp_i and isinstance(Disp_i, (int, float)):
        Disp_f = st.number_input("Digite o valor final para Dispersão [ps/nm/km] (Digite valores entre 12,75 e 17,75): ", key="Disp_f")
        if Disp_f and isinstance(Disp_f, (int, float)):
            p = st.number_input("Digite o valor do incremento para Dispersão (Digite valores maiores ou igual a 0,2): ", key="p")
            if p and isinstance(p, (int, float)):
                if st.button("Calcular Previsão"):
                    fig_max_q, input_values_max_q, predictions_max_q = fazer_previsao_max_q(Att, Disp_i, Disp_f, p, model_max_q, model_type)
                    fig_min_ber, input_values_min_ber, predictions_min_ber = fazer_previsao_min_ber(Att, Disp_i, Disp_f, p, model_min_ber, model_type)

st.markdown("<hr/>", unsafe_allow_html=True)

# Seção de visualização
col1, col2 = st.columns(2)

with col1:
    if fig_max_q is not None:
        st.pyplot(fig_max_q)

with col2:
    if fig_min_ber is not None:
        st.pyplot(fig_min_ber)

st.markdown("<hr/>", unsafe_allow_html=True)

# Condição para exibir a frase 'Dados de Entrada e Previsões'
if input_values_max_q is not None and input_values_min_ber is not None and predictions_max_q is not None and predictions_min_ber is not None:
    st.subheader('Dados de Entrada e Previsões')

    # Formate os números para 8 casas decimais
    data = pd.DataFrame({
        'Att [dB/km]': input_values_max_q['Att'],
        'Disp [ps/nm/km]': input_values_max_q['Disp'],
        'MAX_Q_FACTOR': ["%.8f" % val for val in predictions_max_q],
        'MIN_BER': ["%.8f" % val for val in predictions_min_ber]
    })

    # Exibir a tabela
    st.table(data)

    # Criar um diretório temporário para armazenar o arquivo Excel
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, "previsoes_opticas.xlsx")

    # Escrever os dados na planilha usando xlsxwriter
    workbook = xlsxwriter.Workbook(temp_file_path)
    worksheet = workbook.add_worksheet()

    for i, col in enumerate(data.columns):
        worksheet.write(0, i, col)
        for j, val in enumerate(data[col]):
            worksheet.write(j + 1, i, val)

    workbook.close()

    # Link para download do arquivo Excel
    st.markdown('## Download Tabela XLSX')
    st.write("Clique no link abaixo para baixar a tabela Excel:")
    with open(temp_file_path, 'rb') as f:
        xls_data = f.read()
        b64 = base64.b64encode(xls_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="previsoes_opticas.xlsx">Baixar Tabela Excel</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Limpar o diretório temporário quando não for mais necessário
    temp_dir.cleanup()