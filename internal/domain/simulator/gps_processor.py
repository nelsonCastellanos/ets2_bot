from abc import ABC, abstractmethod
import cv2
import numpy as np

class GPSImageProcessor(ABC):
    @abstractmethod
    def process_gps_image(self, image):
        """
        Procesa la imagen del GPS y retorna la dirección deseada (en grados) 
        si se detecta, o None en caso contrario.
        """
        pass

class GPSImageProcessorImpl(GPSImageProcessor):
    def process_gps_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50, 
                                minLineLength=30, maxLineGap=10)

        angle = None

        if lines is not None:
            longest_line = None
            max_length = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.hypot(x2 - x1, y2 - y1)
                if length > max_length:
                    max_length = length
                    longest_line = (x1, y1, x2, y2)

            if longest_line:
                x1, y1, x2, y2 = longest_line
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle = np.degrees(angle_rad)
                if angle < -45:
                    angle = 90 + angle
        return angle

def debug_gps_image(image_path, output_path="debug_output.png"):
    image = cv2.imread(image_path)
    if image is None:
        print("No se pudo cargar la imagen.")
        return

    processor = GPSImageProcessorImpl()
    angle = processor.process_gps_image(image)

    if angle is not None:
        text = f"Angle: {angle:.2f} deg"
        color = (0, 255, 0)
    else:
        text = "No GPS detected"
        color = (0, 0, 255)

    cv2.putText(image, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imwrite(output_path, image)
    cv2.imshow("Debug GPS", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()