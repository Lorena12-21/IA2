import numpy as np
import cv2

print("=" * 60)
print("TALLER DE LABORATORIO 1: BARS Y BINARIZACIÓN MANUAL")
print("=" * 60)

# 1. Matriz de prueba 4x4 en escala de grises
matriz_grises = np.array([
    [ 50, 120, 200,  15],
    [180,  90,  10, 240],
    [130,  75, 220,  80],
    [ 95, 160, 110, 190]
], dtype=np.uint8)

umbral_T = 100

# 2. Aplicar Binarización Manual (If/Else equivalente con NumPy)
matriz_binaria = np.where(matriz_grises >= umbral_T, 255, 0).astype(np.uint8)

print("\nMatriz Original (Grises):")
print(matriz_grises)
print(f"\nMatriz Binarizada (Umbral T = {umbral_T}):")
print(matriz_binaria)


print("\n" + "=" * 60)
print("TALLER DE LABORATORIO 2: UMBRALIZACIÓN ADAPTATIVA Y OTSU")
print("=" * 60)

# 1. Crear imagen sintética con gradiente de iluminación
np.random.seed(42)
imagen_gradiente = np.linspace(0, 255, 256).reshape(16, 16).astype(np.uint8)

# 2. Umbralización Global Tradicional
_, otsu_thresh = cv2.threshold(imagen_gradiente, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Umbralización Adaptativa (Gausiana)
adaptive_thresh = cv2.adaptiveThreshold(
    imagen_gradiente, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 5, 2
)

print("\nUmbralización Otsu (Global) procesada.")
print("Umbralización Adaptativa procesada correctamente frente a cambios de iluminación.")