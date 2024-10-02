# Importar librerías necesarias
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Paso 1: Crear un dataset de clasificación simulado
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Paso 2: Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 3: Definir los modelos base
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Paso 4: Crear un ensemble combinando ambos modelos
ensemble_model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

# Paso 5: Ajustar el modelo ensemble con los datos de entrenamiento
ensemble_model.fit(X_train, y_train)

# Paso 6: Hacer predicciones sobre los datos de prueba
y_pred_ensemble = ensemble_model.predict(X_test)

# Paso 7: Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f"Accuracy with Ensemble: {accuracy}")
