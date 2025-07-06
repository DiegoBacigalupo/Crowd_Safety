from ultralytics import YOLO
import cv2
from deep_sort_realtime.deep_sort.deepsort_tracker import DeepSort

# Cargar el modelo YOLO
model = YOLO("best.pt")

# Inicializar el tracker DeepSORT
deepsort = DeepSort()

# Abrir el video ya grabado
cap = cv2.VideoCapture("C:\\Users\\Diegote\\OneDrive\\Imágenes\\Escritorio\\Proyecto_respira\\pythonProject\\video\\selection.mov")

while cap.isOpened():
    ret, frame = cap.read()  # Leer un fotograma del video

    if not ret:
        break  # Si no se pudo leer el fotograma, salir del bucle

    # Realizar predicción con el modelo YOLO
    resultados = model.predict(frame, imgsz=1280, conf=0.4)  # Cambiar imgsz a 1280 para mayor resolución

    # Extraer las predicciones (coordenadas de las cajas delimitadoras y la confianza)
    detections = []

    for result in resultados[0].boxes:
        # Si la clase detectada es una persona (ID de clase 0 en COCO)
        if result.cls == 0:  # "0" es la clase de "persona"
            x1, y1, x2, y2 = result.xyxy[0].tolist()  # Coordenadas de la caja delimitadora
            conf = float(result.conf)  # Confianza como float
            class_id = int(result.cls)  # Clase como entero

            # Asegurarnos de que detections tiene el formato correcto
            # DeepSORT espera una lista con: [x1, y1, x2, y2, conf, class_id]
            detection = [x1, y1, x2, y2, conf, class_id]
            detections.append(detection)

    # Usar DeepSORT para hacer el seguimiento de las personas con detecciones completas
    if detections:
        tracks = deepsort.update_tracks(detections, frame)

        # Dibujar las cajas de las personas y sus IDs de seguimiento
        for track in tracks:
            bbox = track[:4]  # Coordenadas de la caja
            track_id = int(track[4])  # ID del seguimiento
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame con las anotaciones
    cv2.imshow("DETECCION Y SEGMENTACION", frame)

    # Salir si presionas la tecla 'Esc'
    if cv2.waitKey(1) == 27:
        break

# Liberar el video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()