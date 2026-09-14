import numpy as np
import cv2

print("=" * 60)
print("TALLER DE LABORATORIO 1: FILTRO DE SUAVIZADO Y CONVOLUCIÓN")
print("=" * 60)

# 1. Crear matriz de imagen 5x5 con ruido de tipo sal y pimienta
imagen_ruido = np.array([
    [100, 100, 255, 100, 100],
    [100,   0, 100, 100, 100],
    [100, 100, 100, 255, 100],
    [  0, 100, 100, 100, 100],
    [100, 100, 100, 100,   0]
], dtype=np.uint8)

# 2. Aplicar filtro de Promedio (Blur) con kernel 3x3
filtro_promedio = cv2.blur(imagen_ruido, (3, 3))

# 3. Aplicar filtro Mediana (ideal para ruido sal y pimienta)
filtro_mediana = cv2.medianBlur(imagen_ruido, 3)

print("\nImagen Original con Ruido (5x5):")
print(imagen_ruido)
print("\nResultado Filtro Promedio (Suavizado):")
print(filtro_promedio)
print("\nResultado Filtro Mediana (Eliminación de Ruido):")
print(filtro_mediana)


print("\n" + "=" * 60)
print("TALLER DE LABORATORIO 2: REALCE Y ENFOQUE (SHARPENING)")
print("=" * 60)

# Kernel Laplaciano para realce de bordes/detalles
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

imagen_enrocada = cv2.filter2D(imagen_ruido, -1, kernel_sharpen)

print("\nImagen Enfocada (Sharpening aplicado):")
print(imagen_enrocada)