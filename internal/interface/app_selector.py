import tkinter as tk
from tkinter import ttk

import cv2
import pygetwindow as gw
from PIL import Image, ImageTk

from internal.infrastructure.window_capture import WindowCapture


def list_window_titles():
    """Devuelve una lista de títulos de ventanas activas (no vacíos)."""
    return [w for w in gw.getAllTitles() if w.strip()]


class AppSelector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Selector de Aplicación")
        self.geometry("900x700")
        self.window_capture = WindowCapture()
        self.selected_window_title = None
        self.create_widgets()
        self.update_thumbnails()  # Actualiza miniaturas periódicamente

    def create_widgets(self):
        # Marco para las miniaturas de ventanas
        self.frame_thumbnails = ttk.Frame(self)
        self.frame_thumbnails.pack(side="top", fill="x", padx=5, pady=5)

        # Marco para la vista previa ampliada de la ventana seleccionada
        self.frame_preview = ttk.Frame(self)
        self.frame_preview.pack(side="bottom", fill="both", expand=True, padx=5, pady=5)
        self.label_preview = ttk.Label(self.frame_preview)
        self.label_preview.pack(fill="both", expand=True)

        self.thumbnail_buttons = []

    def update_thumbnails(self):
        # Limpia miniaturas previas
        for btn in self.thumbnail_buttons:
            btn.destroy()
        self.thumbnail_buttons = []

        titles = list_window_titles()
        for title in titles:
            wc = WindowCapture()
            try:
                wc.select_window()
                wc.window = gw.getWindowsWithTitle(title)[0]
                thumb = wc.capture_thumbnail((200, 150))
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(thumb_rgb)
                photo = ImageTk.PhotoImage(pil_img)
                btn = tk.Button(self.frame_thumbnails, image=photo,
                                command=lambda t=title: self.select_window(t))
                btn.image = photo  # Conservar la referencia
                btn.pack(side="left", padx=5, pady=5)
                self.thumbnail_buttons.append(btn)
            except Exception as e:
                print(f"Error capturando ventana '{title}': {e}")

        self.after(5000, self.update_thumbnails)

    def select_window(self, title):
        self.selected_window_title = title
        self.window_capture.select_window(title)
        print(f"Ventana seleccionada: {title}")

    def update_preview(self):
        if self.selected_window_title:
            try:
                frame = self.window_capture.capture()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                preview_width = self.label_preview.winfo_width() or 800
                preview_height = self.label_preview.winfo_height() or 600
                pil_img = pil_img.resize((preview_width, preview_height))
                photo = ImageTk.PhotoImage(pil_img)
                self.label_preview.configure(image=photo)
                self.label_preview.image = photo
            except Exception as e:
                print(f"Error actualizando vista previa: {e}")
        self.after(100, self.update_preview)


def main():
    app = AppSelector()
    app.update_preview()
    app.mainloop()


if __name__ == "__main__":
    main()
