from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, ImageOps
import numpy as np
import threading

model = load_model('Best_points.h5')

def predict_letter(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

def get_letter(num):
    letter = chr(65+num) # 65 == char value of 'A', so class 0=A, class 1=B, etc.
    return letter

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0
        
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        # self.classify_btn = tk.Button(self, text = "Recognise", command = self.update_prediction)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear)

        # Grid structure
        self.canvas.grid(row=1, column=0, pady=2, sticky=W)
        self.label.grid(row=2, column=0,pady=2, padx=2)
        # self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=0, column=0, pady=2)
        
        self.canvas.bind("<B1-Motion>", self.draw)

        threading.Timer(1.0, self.update_prediction).start()

    def clear(self):
        self.canvas.delete("all")
        
    def update_prediction(self):
        HWND = self.canvas.winfo_id() 
        rect = win32gui.GetWindowRect(HWND)
        a,b,c,d = rect
        rect=(a+4,b+4,c-4,d-4)
        im = ImageGrab.grab(rect)

        digit, acc = predict_letter(im)
        self.label.configure(text= get_letter(digit)+', '+ str(int(acc*100))+'%')
        threading.Timer(1.0, self.update_prediction).start()

    def draw(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()