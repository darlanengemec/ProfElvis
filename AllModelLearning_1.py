# Importar as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

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

models = []
train_sizes = []

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

  # Adicionar modelos e proporções à lista
  models.append({'DecisionTreeRegressor': dtr, 'RandomForestRegressor': rfr, 'GradientBoostingRegressor': gbr})
  train_sizes.append(train_size)

  # Fazer as previsões com os dados de teste
  y_pred_dtr = dtr.predict(X_test)
  y_pred_rfr = rfr.predict(X_test)
  y_pred_gbr = gbr.predict(X_test)

  # Calcular o coeficiente de determinação (R^2) entre os valores reais e previstos
  r2_dtr = r2_score(y_test, y_pred_dtr)
  r2_rfr = r2_score(y_test, y_pred_rfr)
  r2_gbr = r2_score(y_test, y_pred_gbr)

  
  # Adicionar o valor de R^2 à lista
  r2_dtr_list.append(r2_dtr)
  r2_rfr_list.append(r2_rfr)
  r2_gbr_list.append(r2_gbr)

# Criar um gráfico de R^2 versus porcentagem da base de treinamento
plt.plot(np.arange(0.1, 1, 0.1), r2_dtr_list, marker="o", label='Decision Tree Regressor')
plt.plot(np.arange(0.1, 1, 0.1), r2_rfr_list, marker="o", label='Random Forest Regression')
plt.plot(np.arange(0.1, 1, 0.1), r2_gbr_list, marker="o", label='Gradient Boosting Regressor')
#plt.plot(rshb.Treinamento /100, rshb.r2, marker="o", label='Histogram Gradient Boosting Regressor')
plt.legend()
# plt.text(0.5, -0.55, 'Min_BER', dict(size=25))
plt.xlabel("Base de Treinamento (%)")
plt.ylabel("R^2")
plt.title("Modelos de Machine Learning (MIN_BER)")
plt.show()

# Salvar modelos treinados em um arquivo
joblib.dump((models, train_sizes), 'trained_models_MIN_BER.pkl')