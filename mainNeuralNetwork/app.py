import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.withdraw()

imagePath=r"digit.png"
img=Image.open(imagePath).convert('L')
img = img.resize((28, 28))
img = np.asarray(img)
img = img / 255.0
img = img.reshape(784, 1)


param=np.load(r"data\mnist.npz")
w1=param["w1"]
w2=param["w2"]
b1=param["b1"]
b2=param["b2"]

def ReLU(z):
    return np.maximum(0,z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forwardP(w1,b1,w2,b2,x):
    z1=w1.dot(x)+b1
    a1=ReLU(z1)
    z2=w2.dot(a1)+b2
    a2=softmax(z2)
    return z1,a1,z2,a2

def predict_image(Img):
    _, _, _, a2 = forwardP(w1, b1, w2, b2, Img)

    prediction=int(np.argmax(a2, axis=0)[0])
    confidence=float(a2[prediction][0])

    messagebox.showinfo(
        "Digit Prediction",
        f"Prediction: {prediction}\nConfidence: {confidence:.2%}"
    )
    
while True:
    input("Save digit.png, then press Enter to identify. Type Ctrl+C to stop.")
    
    img=Image.open(imagePath).convert("L")
    img=img.resize((28, 28))
    img=np.asarray(img)
    img=img / 255.0
    img=img.reshape(784, 1)

    predict_image(img)
