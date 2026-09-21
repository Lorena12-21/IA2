import numpy as np

print("=" * 60)
print("SESIÓN 11: PERCEPTRÓN - COMPUERTA LÓGICA OR")
print("=" * 60)

# 1. Función de Activación Escalón
def funcion_escalon(z):
    return 1 if z >= 0 else 0

# 2. Estructura del Perceptrón
def perceptron(X, W, b):
    Z = np.dot(X, W) + b
    return funcion_escalon(Z)

# 3. Pesos y Sesgo ajustados para la compuerta OR
# Con W = [0.5, 0.5] y b = -0.2:
# [0, 0] -> 0*0.5 + 0*0.5 - 0.2 = -0.2 (< 0) -> Salida: 0
# [1, 0] -> 1*0.5 + 0*0.5 - 0.2 =  0.3 (>= 0) -> Salida: 1
# [0, 1] -> 0*0.5 + 1*0.5 - 0.2 =  0.3 (>= 0) -> Salida: 1
# [1, 1] -> 1*0.5 + 1*0.5 - 0.2 =  0.8 (>= 0) -> Salida: 1
pesos_or = np.array([0.5, 0.5])
sesgo_or = -0.2

entradas = [
    np.array([0, 0]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])
]

print("\n--- PRUEBA COMPUERTA OR ---")
for x in entradas:
    res = perceptron(x, pesos_or, sesgo_or)
    print(f"Entrada {x} -> Salida: {res}")
    