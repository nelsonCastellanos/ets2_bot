import pyautogui

class KeyboardController:
    def set_axis(self, axis: int, value: int):
        center = 16383
        tolerance = 1000
        if value < center - tolerance:
            print("Simulando tecla: LEFT")
            pyautogui.press('left')
        elif value > center + tolerance:
            print("Simulando tecla: RIGHT")
            pyautogui.press('right')
        else:
            print("Eje centrado, sin acción de dirección.")

    def press_button(self, button: int):
        print("Simulando tecla: UP (presionado)")
        pyautogui.keyDown('up')

    def release_button(self, button: int):
        print("Simulando tecla: UP (liberado)")
        pyautogui.keyUp('up')
