import numpy as np
from sklearn.svm import SVC

print("=" * 60)
print("SESIÓN 10: ALGORITMO SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)

# 1. Dataset de prueba con un punto no linealmente separable ([5, 5])
X = np.array([
    [2, 2],
    [3, 3],
    [4, 2],
    [6, 6],
    [7, 8],
    [8, 7],
    [5, 5]  # Punto no separable linealmente
])

Y = np.array([0, 0, 0, 1, 1, 1, 0]) # [5,5] pertenece a Clase 0

# 2. Modelo con Kernel Lineal
print("\n--- ENTRENANDO CON KERNEL LINEAL ---")
svm_lineal = SVC(kernel='linear')
svm_lineal.fit(X, Y)
print("Vectores de Soporte (Lineal):\n", svm_lineal.support_vectors_)

punto_prueba = np.array([[5, 4]])
pred_lineal = svm_lineal.predict(punto_prueba)
print("Predicción para [5, 4] (Lineal):", pred_lineal[0])

# 3. Modelo con Kernel RBF (Proyección no lineal)
print("\n--- ENTRENANDO CON KERNEL RBF ---")
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X, Y)
print("Vectores de Soporte (RBF):\n", svm_rbf.support_vectors_)

pred_rbf = svm_rbf.predict(punto_prueba)
print("Predicción para [5, 4] (RBF):", pred_rbf[0])

print("\n--- REFLEXIÓN: ESCENARIOS EN EL MUNDO REAL ---")
print("Un kernel lineal falla cuando hay patrones concéntricos o intercalados.")
print("Ejemplo práctico: En diagnóstico médico, los biomarcadores de una enfermedad")
print("suelen estar concentrados dentro de un rango específico rodeado por valores normales.")