import numpy as np
from sklearn.neighbors import KNeighborsClassifier

print("=" * 60)
print("SESIÓN 9: ALGORITMO DE K-VECINOS MÁS CERCANOS (KNN)")
print("=" * 60)

# 1. Dataset Ampliado (Edad, Salario en miles, N° Hijos)
X_entrenamiento = np.array([
    [20, 30, 0], # Cliente 1: A
    [40, 50, 2], # Cliente 2: B
    [35, 45, 1], # Cliente 3: C
    [22, 25, 0], # Cliente 4
    [48, 65, 3], # Cliente 5
    [52, 70, 2], # Cliente 6
    [25, 38, 0], # Cliente 7
    [31, 42, 1], # Cliente 8
    [45, 60, 2], # Cliente 9
    [29, 33, 1]  # Cliente 10
])

# Etiquetas: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])

# 2. Experimentos con K=1 y K=5
nuevo_cliente = np.array([[30, 40, 1]])

print("\n--- EXPERIMENTO CON K = 1 ---")
modelo_k1 = KNeighborsClassifier(n_neighbors=1)
modelo_k1.fit(X_entrenamiento, Y_entrenamiento)
pred_k1 = modelo_k1.predict(nuevo_cliente)
print("Clase predicha con K=1:", "COMPRA" if pred_k1[0] == 1 else "NO COMPRA")

print("\n--- EXPERIMENTO CON K = 5 ---")
modelo_k5 = KNeighborsClassifier(n_neighbors=5)
modelo_k5.fit(X_entrenamiento, Y_entrenamiento)
pred_k5 = modelo_k5.predict(nuevo_cliente)
print("Clase predicha con K=5:", "COMPRA" if pred_k5[0] == 1 else "NO COMPRA")

print("\n--- REFLEXIÓN: LA MALDICIÓN DE LA DIMENSIONALIDAD ---")
print("Si aumentamos a 1,000 dimensiones, la distancia euclidiana tiende a concentrarse")
print("y volverse casi idéntica entre todos los puntos del espacio, haciendo que el")
print("concepto de 'vecino cercano' pierda sentido y requiera un volumen exponencial de datos.")