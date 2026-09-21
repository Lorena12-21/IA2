import numpy as np

print("=" * 60)
print("SESIÓN 12: RED NEURONAL MULTICAPA (MLP) EN BATCH")
print("=" * 60)

# Función de Activación Sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# 1. ENTRADA (X): Batch de 2 clientes simultáneos (Matriz 2x3)
X = np.array([
    [0.5, 0.8, 0.2],  # Cliente 1
    [0.1, 0.9, 0.9]   # Cliente 2
])

# 2. CAPA OCULTA (4 Neuronas)
W1 = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [-0.5, 0.6, 0.7, -0.8],
    [0.9, -0.1, 0.2, 0.3]
])
b1 = np.array([0.1, 0.2, 0.3, 0.4])

# Proceso Capa Oculta
Z1 = np.dot(X, W1) + b1
A1 = sigmoide(Z1)

# 3. CAPA DE SALIDA (1 Neurona)
W2 = np.array([0.5, 0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# Proceso Capa Final
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

print("\n--- RESULTADOS DEL BATCH ---")
print("Valores Z1 (Combinaciones lineales puras):\n", np.round(Z1, 4))
print("\nValores A1 (Activaciones tras Sigmoide entre 0 y 1):\n", np.round(A1, 4))
print("\nPredicción de Salida para Cliente 1:", np.round(Salida_Final[0], 4))
print("Predicción de Salida para Cliente 2:", np.round(Salida_Final[1], 4))
