# Importar as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Carregar a base de dados
df = pd.read_csv("database.csv")


# Separar as variáveis independentes e dependente
modelo_features=["Att","Disp"]
X = df[modelo_features]

y = df.Min_BER

# Criar uma lista vazia para armazenar os valores de R^2

r2_dtr_list = []
r2_rfr_list = []
r2_gbr_list = []

mae_dtr_list = []
mae_rfr_list = []
mae_gbr_list = []

mse_dtr_list = []
mse_rfr_list = []
mse_gbr_list = []

# Criar um loop para variar a porcentagem da base de treinamento de 10% a 90%
for i in range(1, 10):
  # Dividir a base em treinamento e teste
  train_size = i * 0.1 # Porcentagem da base de treinamento
  test_size = 1 - train_size # Porcentagem da base de teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

  # Treinar o modelo de Gradient Boosting Regressor com os dados de treinamento
  dtr = DecisionTreeRegressor(random_state=1)
  rfr = RandomForestRegressor(n_estimators=100, random_state=0)
  gbr = GradientBoostingRegressor(random_state=42)

  dtr.fit(X_train, y_train)
  rfr.fit(X_train, y_train)
  gbr.fit(X_train, y_train)

  # Fazer as previsões com os dados de teste
  y_pred_dtr = dtr.predict(X_test)
  y_pred_rfr = rfr.predict(X_test)
  y_pred_gbr = gbr.predict(X_test)

  # Calcular o coeficiente de determinação (R^2 , MAE e MSE) entre os valores reais e previstos
  r2_dtr = r2_score(y_test, y_pred_dtr)
  r2_rfr = r2_score(y_test, y_pred_rfr)
  r2_gbr = r2_score(y_test, y_pred_gbr)

  mae_dtr=mean_absolute_error(y_test,y_pred_dtr)
  mae_rfr=mean_absolute_error(y_test,y_pred_rfr)
  mae_gbr=mean_absolute_error(y_test,y_pred_gbr)

  mse_dtr=mean_squared_error(y_test,y_pred_dtr)
  mse_rfr=mean_squared_error(y_test,y_pred_rfr)
  mse_gbr=mean_squared_error(y_test,y_pred_gbr)

  
  # Adicionar o valor de R^2, MAE e MSE à lista
  r2_dtr_list.append(r2_dtr)
  r2_rfr_list.append(r2_rfr)
  r2_gbr_list.append(r2_gbr)

  mae_dtr_list.append(mae_dtr)
  mae_rfr_list.append(mae_rfr)
  mae_gbr_list.append(mae_gbr)

  mse_dtr_list.append(mse_dtr)
  mse_rfr_list.append(mse_rfr)
  mse_gbr_list.append(mse_gbr)

porcentagem=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Nome do arquivo CSV
nome_arquivo = "Valores_Min_BER.csv"

# Abrir o arquivo CSV para escrita
with open(nome_arquivo, mode='w', newline='') as arquivo_csv: 
# Criar um objeto de escrita CSV
    escritor = csv.writer(arquivo_csv)

  # Escrever os conjuntos de dados no arquivo CSV
    escritor.writerow(["Porcentagem","R^2 DTR", "R^2 RFR", "R^2 GBR", "MAE DTR", "MAE RFR", "MAE GBR","MSE DTR", "MSE RFR", "MSE GBR"])  
    # Escrever cabeçalho
    for i in range(len(r2_dtr_list)):
        escritor.writerow([porcentagem[i], r2_dtr_list[i], r2_rfr_list[i], r2_gbr_list[i], mae_dtr_list[i], mae_rfr_list[i], mae_gbr_list[i], mse_dtr_list[i], mse_rfr_list[i], mse_gbr_list[i]])

print("Conjuntos de dados salvos com sucesso no arquivo CSV.")

