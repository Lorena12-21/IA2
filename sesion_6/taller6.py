import numpy as np
import cv2

print("=" * 60)
print("TALLER DE LABORATORIO 1: EXTRACCIÓN Y DIBUJO DE CONTORNOS")
print("=" * 60)

# 1. Crear una imagen binaria 100x100 con un cuadrado blanco centrado
imagen_cuadrado = np.zeros((100, 100), dtype=np.uint8)
imagen_cuadrado[25:75, 25:75] = 255

# 2. Encontrar contornos en la imagen binaria
contornos, jerarquia = cv2.findContours(
    imagen_cuadrado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print(f"Número de contornos detectados: {len(contornos)}")

print("\n" + "=" * 60)
print("TALLER DE LABORATORIO 2: DESCRIPTORES DE FORMA Y MOMENTOS")
print("=" * 60)

if len(contornos) > 0:
    cnt = contornos[0]
    
    # 1. Calcular Área y Perímetro
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    
    # 2. Calcular Bounding Box (Caja Delimitadora)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 3. Calcular Momentos y Centroide (Cx, Cy)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
        
    print(f"Área del contorno: {area:.2f} píxeles")
    print(f"Perímetro del contorno: {perimetro:.2f} píxeles")
    print(f"Caja delimitadora: x={x}, y={y}, ancho={w}, alto={h}")
    print(f"Centroide de la figura: (Cx={cx}, Cy={cy})")
    