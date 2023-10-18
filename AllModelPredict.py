# Importar as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=42)

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

  # Calcular o coeficiente de determinação (R^2) entre os valores reais e previstos
r2_dtr = r2_score(y_test, y_pred_dtr)
r2_rfr = r2_score(y_test, y_pred_rfr)
r2_gbr = r2_score(y_test, y_pred_gbr)


prediction_dtr=dtr.predict(X)
prediction_rfr=rfr.predict(X)
prediction_gbr=gbr.predict(X)

mae_dtr=mean_absolute_error(y,prediction_dtr)
mae_rfr=mean_absolute_error(y,prediction_rfr)
mae_gbr=mean_absolute_error(y,prediction_gbr)

mse_dtr=mean_squared_error(y,prediction_dtr)
mse_rfr=mean_squared_error(y,prediction_rfr)
mse_gbr=mean_squared_error(y,prediction_gbr)


new_data_frame=df[['Att','Disp','Min_BER']]
new_data_frame['Pred_DTR']=prediction_dtr
new_data_frame['Pred_RFR']=prediction_rfr
new_data_frame['Pred_GBR']=prediction_gbr

new_data_frame['mae_DTR']=mae_dtr
new_data_frame['mae_RFR']=mae_rfr
new_data_frame['mae_GBR']=mae_gbr

new_data_frame['mse_DTR']=mse_dtr
new_data_frame['mse_RFR']=mse_rfr
new_data_frame['mse_GBR']=mse_gbr

new_data_frame.to_csv('predict_values_Min_BER.csv', index=False)


