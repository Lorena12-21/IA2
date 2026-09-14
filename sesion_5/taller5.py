import numpy as np
import cv2

print("=" * 60)
print("TALLER DE LABORATORIO 1: FILTROS DE DERIVADA (SOBEL Y LAPLACIANO)")
print("=" * 60)

# 1. Crear una matriz de prueba 5x5 con una transición brusca de contraste (borde vertical)
imagen_borde = np.array([
    [ 10,  10, 200, 200, 200],
    [ 10,  10, 200, 200, 200],
    [ 10,  10, 200, 200, 200],
    [ 10,  10, 200, 200, 200],
    [ 10,  10, 200, 200, 200]
], dtype=np.uint8)

# 2. Aplicar el operador Sobel en las direcciones X e Y
sobel_x = cv2.Sobel(imagen_borde, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagen_borde, cv2.CV_64F, 0, 1, ksize=3)

# Magnitud del gradiente aproximada
magnitud_sobel = np.hypot(sobel_x, sobel_y)

# 3. Aplicar el filtro Laplaciano
laplaciano = cv2.Laplacian(imagen_borde, cv2.CV_64F)

print("\nImagen Original (Borde Vertical):")
print(imagen_borde)
print("\nSobel X (Bordes Verticales Detectados):")
print(np.abs(sobel_x).astype(np.uint8))
print("\nMagnitud del Gradiente Sobel:")
print(magnitud_sobel.astype(np.uint8))


print("\n" + "=" * 60)
print("TALLER DE LABORATORIO 2: ALGORITMO CANNY")
print("=" * 60)

# Aplicar detector Canny con umbrales minVal=50 y maxVal=150
bordes_canny = cv2.Canny(imagen_borde, 50, 150)

print("\nBordes detectados por el algoritmo Canny:")
print(bordes_canny)