# Importar as bibliotecas necessárias
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Carregar a base de dados
df = pd.read_csv("database.csv")


# Separar as variáveis independentes e dependente
modelo_features=["Att","Disp"]
X = df[modelo_features]

y = df.Min_BER

# Criar um dicionário para armazenar os modelos treinados
model_dict = {}

# Criar um loop para variar a porcentagem da base de treinamento de 10% a 90%
for i in range(1, 10):
    # Dividir a base em treinamento e teste
    train_size = i * 0.1  # Porcentagem da base de treinamento
    test_size = 1 - train_size  # Porcentagem da base de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

    # Treinar o modelo de Gradient Boosting Regressor com os dados de treinamento
    dtr = DecisionTreeRegressor(random_state=1)
    rfr = RandomForestRegressor(n_estimators=100, random_state=0)
    gbr = GradientBoostingRegressor(random_state=42)

    dtr.fit(X_train, y_train)
    rfr.fit(X_train, y_train)
    gbr.fit(X_train, y_train)

    # Armazenar os modelos treinados em uma lista
    modelos_treinados = [dtr, rfr, gbr]

    # Armazenar a lista de modelos no dicionário com IDs específicos
    for modelo, modelo_nome in zip(modelos_treinados, ["dtr", "rfr", "gbr"]):
        model_dict[f"{modelo_nome}_{i * 10}"] = modelo

# Print dos IDs
    print(f"IDs para proporção de {i * 10}%:")
    for modelo_nome in ["dtr", "rfr", "gbr"]:
        print(f"{modelo_nome.capitalize()} ID: {modelo_nome}_{i * 10}")

# Salvar o dicionário com os modelos em um arquivo pkl
with open('trained_models_MIN_BER.pkl', 'wb') as file:
    pickle.dump(model_dict, file)
# Carregar o dicionário de modelos para verificar o que foi salvo
with open('trained_models_MIN_BER.pkl', 'rb') as file:
    saved_model_dict = pickle.load(file)

print("Conteúdo do arquivo pkl após salvamento:")
print(saved_model_dict)