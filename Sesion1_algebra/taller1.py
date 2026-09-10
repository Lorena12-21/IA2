import numpy as np

print("=" * 60)
print("TALLER DE LABORATORIO 1: TRANSFORMACIONES AFINES")
print("=" * 60)

# 1. Crear matriz de prueba 5x5
np.random.seed(42)
matriz_original = np.random.randint(200, 256, (5, 5), dtype=np.int32)

print("\n1. Matriz Original (Sobreexpuesta 5x5):")
print(matriz_original)

# 2. Reducción de contraste (50%) y disminución de brillo (-50)
alpha = 0.5
beta = -50.0

matriz_procesada_raw = alpha * matriz_original + beta

# 3. Aplicar acotamiento (clipping) y convertir a uint8
matriz_procesada = np.clip(matriz_procesada_raw, 0, 255).astype(np.uint8)

print("\n2. Matriz Procesada (Contraste -50%, Brillo -50):")
print(matriz_procesada)


print("TALLER DE LABORATORIO FINAL: PROGRAMANDO UN KERNEL")

# 1. Definición de la Sección de Imagen (I) y Kernel (K)
imagen_I = np.array([
    [100, 100, 100],
    [100, 200, 100],
    [100, 100, 100]
], dtype=np.float32)

kernel_K = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# 2. Producto Hadamard
producto_hadamard = imagen_I * kernel_K

# 3. Suma de todos los valores (píxel central resultante)
pixel_central = np.sum(producto_hadamard)

print("\nMatriz Imagen (I):\n", imagen_I)
print("\nKernel de Realce (K):\n", kernel_K)
print(f"\nValor calculado para el píxel central: {pixel_central:.2f}")