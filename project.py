from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Crear dataset de ejemplo
X, y = make_classification(n_samples=1000, n_features=10)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Modelo de Regresión Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
