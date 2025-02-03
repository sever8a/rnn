import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Cargar los datos
data = pd.read_csv('transacciones.csv')

# Preprocesar los datos
data = preprocess_data(data)  # Esta función incluye limpieza y transformación de datos

# Dividir en conjuntos de entrenamiento y prueba
X = data.drop('fraude', axis=1)
y = data['fraude']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))

# Guardar el modelo entrenado
import joblib
joblib.dump(modelo, 'modelo_fraude.pkl')
