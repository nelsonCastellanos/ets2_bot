import numpy as np

from internal.domain.simulator.gps_processor import GPSImageProcessor
from internal.domain.simulator.repository_boundary import ScreenCapture


class DataLogger:
    def __init__(self,
                 screen_capture: ScreenCapture,
                 gps_processor: GPSImageProcessor,
                 output_dir='training_data', csv_filename='labels.csv'):
        """
        screen_capture: Adaptador que implementa ScreenCapture.
        gps_processor: Implementación que procesa la imagen del GPS y retorna la dirección si se detecta.
        """
        self.screen_capture = screen_capture
        self.gps_processor = gps_processor
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, self.csv_filename)
        self.listener = None
        self.current_keys = set()
        self.previous_frame = None  # Para detectar movimiento

        self.csv_file = open(self.csv_path, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if os.stat(self.csv_path).st_size == 0:
            self.csv_writer.writerow(['image_filename', 'keys'])

    def on_press(self, key):
        try:
            self.current_keys.add(key.char)
        except AttributeError:
            self.current_keys.add(str(key))

    def on_release(self, key):
        try:
            self.current_keys.discard(key.char)
        except AttributeError:
            self.current_keys.discard(str(key))

    def start_key_listener(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def stop_key_listener(self):
        if self.listener:
            self.listener.stop()
            self.listener.join()

    def gps_present(self, frame: np.ndarray) -> bool:
        """
        Recorta la región del GPS y usa el procesador para determinar si hay una dirección visible.
        """
        height, width, _ = frame.shape
        gps_region = {
            "top": int(height * 0.7),
            "left": int(width * 0.8),
            "width": int(width * 0.2),
            "height": int(height * 0.2)
        }
        # Recortar la región donde se espera ver el GPS.
        gps_img = frame[
                  gps_region["top"]:gps_region["top"] + gps_region["height"],
                  gps_region["left"]:gps_region["left"] + gps_region["width"]
                  ]
        angle = self.gps_processor.process_gps_image(gps_img)
        if angle is not None:
            print(f"GPS detectado con ángulo: {angle:.2f}°")
            return True
        else:
            print("GPS no detectado en la imagen.")
            return False

    def vehicle_in_motion(self, frame: np.ndarray) -> bool:
        """
        Detecta movimiento comparando la imagen actual con la anterior.
        Se utiliza la diferencia promedio en escala de grises para determinar si hay cambios significativos.
        """
        if self.previous_frame is None:
            self.previous_frame = frame
            # La primera imagen no permite comparar, asumimos que no hay movimiento.
            print("No hay datos previos para comparar movimiento.")
            return False

        # Calcular la diferencia absoluta entre ambas imágenes.
        diff = cv2.absdiff(self.previous_frame, frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mean_diff = np.mean(gray)
        self.previous_frame = frame  # Actualizar para la siguiente comparación

        # Definir un umbral (este valor se puede ajustar según la sensibilidad deseada)
        movement_threshold = 5.0
        if mean_diff > movement_threshold:
            print(f"Movimiento detectado (diferencia promedio: {mean_diff:.2f}).")
            return True
        else:
            print(f"Sin movimiento significativo (diferencia promedio: {mean_diff:.2f}).")
            return False

    def capture_and_log(self, num_frames=100, delay=0.1):
        frame_count = 0
        while frame_count < num_frames:
            # Capturamos la imagen completa usando el adaptador inyectado.
            frame = self.screen_capture.capture()
            if frame is None:
                print("No se pudo capturar la imagen. Se intenta nuevamente...")
                time.sleep(delay)
                continue

            # Solo se guarda si se detecta el GPS y el vehículo está en movimiento.
            if not self.gps_present(frame):
                print("Esperando a que el GPS sea visible...")
                time.sleep(delay)
                continue

            if not self.vehicle_in_motion(frame):
                print("El vehículo no se encuentra en movimiento. No se registrará la captura.")
                time.sleep(delay)
                continue

            # Guardar la imagen en disco.
            image_filename = f"frame_{int(time.time() * 1000)}.png"
            image_path = os.path.join(self.output_dir, image_filename)
            cv2.imwrite(image_path, frame)

            # Registrar las teclas presionadas junto con el nombre de la imagen en el CSV.
            keys_str = ' '.join(sorted(self.current_keys))
            self.csv_writer.writerow([image_filename, keys_str])
            self.csv_file.flush()

            print(f"Guardado {image_filename} con teclas: {keys_str}")
            frame_count += 1
            time.sleep(delay)

    def close(self):
        self.csv_file.close()


# internal/infrastructure/data_logger.py

import csv
import os
import time
import cv2
import numpy as np
from pynput import keyboard
from internal.domain.simulator.repository_boundary import ScreenCapture
from internal.domain.simulator.gps_processor import GPSImageProcessor


class DataLogger:
    def __init__(self, screen_capture: ScreenCapture, gps_processor: GPSImageProcessor,
                 output_dir='../training_data', csv_filename='labels.csv'):
        """
        screen_capture: Adaptador que implementa ScreenCapture.
        gps_processor: Implementación que procesa la imagen del GPS y retorna la dirección si se detecta.
        """
        self.screen_capture = screen_capture
        self.gps_processor = gps_processor
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, self.csv_filename)
        self.listener = None
        self.current_keys = set()
        self.previous_frame = None  # Para detectar movimiento

        self.csv_file = open(self.csv_path, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if os.stat(self.csv_path).st_size == 0:
            self.csv_writer.writerow(['image_filename', 'keys'])

    def on_press(self, key):
        try:
            self.current_keys.add(key.char)
        except AttributeError:
            self.current_keys.add(str(key))

    def on_release(self, key):
        try:
            self.current_keys.discard(key.char)
        except AttributeError:
            self.current_keys.discard(str(key))

    def start_key_listener(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def stop_key_listener(self):
        if self.listener:
            self.listener.stop()
            self.listener.join()

    def gps_present(self, frame: np.ndarray) -> bool:
        """
        Recorta la región del GPS y usa el procesador para determinar si hay una dirección visible.
        """
        height, width, _ = frame.shape
        gps_region = {
            "top": int(height * 0.7),
            "left": int(width * 0.8),
            "width": int(width * 0.2),
            "height": int(height * 0.2)
        }
        # Recortar la región donde se espera ver el GPS.
        gps_img = frame[
                  gps_region["top"]:gps_region["top"] + gps_region["height"],
                  gps_region["left"]:gps_region["left"] + gps_region["width"]
                  ]
        angle = self.gps_processor.process_gps_image(gps_img)
        if angle is not None:
            print(f"GPS detectado con ángulo: {angle:.2f}°")
            return True
        else:
            print("GPS no detectado en la imagen.")
            return False

    def vehicle_in_motion(self, frame: np.ndarray) -> bool:
        """
        Detecta movimiento comparando la imagen actual con la anterior.
        Se utiliza la diferencia promedio en escala de grises para determinar si hay cambios significativos.
        """
        if self.previous_frame is None:
            self.previous_frame = frame
            # La primera imagen no permite comparar, asumimos que no hay movimiento.
            print("No hay datos previos para comparar movimiento.")
            return False

        # Calcular la diferencia absoluta entre ambas imágenes.
        diff = cv2.absdiff(self.previous_frame, frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mean_diff = np.mean(gray)
        self.previous_frame = frame  # Actualizar para la siguiente comparación

        # Definir un umbral (este valor se puede ajustar según la sensibilidad deseada)
        movement_threshold = 5.0
        if mean_diff > movement_threshold:
            print(f"Movimiento detectado (diferencia promedio: {mean_diff:.2f}).")
            return True
        else:
            print(f"Sin movimiento significativo (diferencia promedio: {mean_diff:.2f}).")
            return False

    def capture_and_log(self, num_frames=100, delay=0.1):
        frame_count = 0
        while frame_count < num_frames:
            # Capturamos la imagen completa usando el adaptador inyectado.
            frame = self.screen_capture.capture()
            if frame is None:
                print("No se pudo capturar la imagen. Se intenta nuevamente...")
                time.sleep(delay)
                continue

            # Solo se guarda si se detecta el GPS y el vehículo está en movimiento.
            if not self.gps_present(frame):
                print("Esperando a que el GPS sea visible...")
                time.sleep(delay)
                continue

            if not self.vehicle_in_motion(frame):
                print("El vehículo no se encuentra en movimiento. No se registrará la captura.")
                time.sleep(delay)
                continue

            # Guardar la imagen en disco.
            image_filename = f"frame_{int(time.time() * 1000)}.png"
            image_path = os.path.join(self.output_dir, image_filename)
            cv2.imwrite(image_path, frame)

            # Registrar las teclas presionadas junto con el nombre de la imagen en el CSV.
            keys_str = ' '.join(sorted(self.current_keys))
            self.csv_writer.writerow([image_filename, keys_str])
            self.csv_file.flush()

            print(f"Guardado {image_filename} con teclas: {keys_str}")
            frame_count += 1
            time.sleep(delay)

    def close(self):
        self.csv_file.close()
