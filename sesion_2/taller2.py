import numpy as np
import cv2

print("=" * 60)
print("TALLER DE LABORATORIO 1: TRANSFORMACIÓN DE ESPACIOS DE COLOR")
print("=" * 60)

# 1. Píxel BGR de prueba completamente amarillo [Azul: 0, Verde: 255, Rojo: 255]
pixel_amarillo_bgr = np.array([0, 255, 255], dtype=np.float32)

# 2. Convertir a escala de grises usando la fórmula ponderada: Y = 0.299*R + 0.587*G + 0.114*B
# En formato BGR de OpenCV/NumPy: [0]=B, [1]=G, [2]=R
valor_gris_calculado = (0.114 * pixel_amarillo_bgr[0]) + (0.587 * pixel_amarillo_bgr[1]) + (0.299 * pixel_amarillo_bgr[2])

print(f"Píxel BGR original (Amarillo): {pixel_amarillo_bgr.astype(int)}")
print(f"Valor en escala de grises calculado: {valor_gris_calculado:.2f} (de 255)")


print("\n" + "=" * 60)
print("TALLER DE LABORATORIO 2: ANÁLISIS ESTADÍSTICO DE HISTOGRAMAS")
print("=" * 60)

# 1. Generar imagen sintética BGR (300x300 píxeles, 3 canales)
np.random.seed(10)
imagen_prueba = np.zeros((300, 300, 3), dtype=np.uint8)
imagen_prueba[:, :, 0] = np.random.randint(50, 150, (300, 300))  # Canal Azul
imagen_prueba[:, :, 1] = np.random.randint(100, 220, (300, 300)) # Canal Verde (Dominante)
imagen_prueba[:, :, 2] = np.random.randint(20, 80, (300, 300))   # Canal Rojo

# 2. Separar los 3 canales BGR
canal_b, canal_g, canal_r = cv2.split(imagen_prueba)

# 3. Calcular histograma para cada canal
hist_b = cv2.calcHist([canal_b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([canal_g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([canal_r], [0], None, [256], [0, 256])

print("Histogramas de los 3 canales calculados exitosamente.")
print("Análisis: El canal Verde presenta mayor concentración en intensidades altas (100-220).")