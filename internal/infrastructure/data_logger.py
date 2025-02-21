import csv
import os
import time
import cv2
from pynput import keyboard
from internal.domain.simulator.repository_boundary import ScreenCapture

class DataLogger:
    def __init__(self, screen_capture: ScreenCapture, output_dir='training_data', csv_filename='labels.csv'):
        """
        screen_capture: Instancia de un adaptador que implementa la interfaz ScreenCapture.
        """
        self.screen_capture = screen_capture
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, self.csv_filename)
        self.listener = None
        self.current_keys = set()

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

    def capture_and_log(self, num_frames=100, delay=0.1):
        frame_count = 0
        while frame_count < num_frames:
            frame = self.screen_capture.capture()
            if frame is None:
                print("No se pudo capturar la imagen. Se intenta nuevamente...")
                time.sleep(delay)
                continue

            image_filename = f"frame_{int(time.time()*1000)}.png"
            image_path = os.path.join(self.output_dir, image_filename)
            cv2.imwrite(image_path, frame)

            keys_str = ' '.join(sorted(self.current_keys))
            self.csv_writer.writerow([image_filename, keys_str])
            self.csv_file.flush()

            print(f"Guardado {image_filename} con teclas: {keys_str}")
            frame_count += 1
            time.sleep(delay)

    def close(self):
        self.csv_file.close()
