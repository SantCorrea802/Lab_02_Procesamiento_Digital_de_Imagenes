import tkinter as tk
from tkinter import filedialog
import cv2
from pathlib import Path
import os
import joblib
from skimage.feature import local_binary_pattern
import numpy as np
from PIL import ImageTk, Image # para poder mostrar la imagen en la ventana de tkinter

IMG_PATH = ""
panel_img = None; # imagen mostrada la inicializamos en None

def loadImgLBP():
    global IMG_PATH, panel_img
  # filedialog.askopenfilename(initialdir='/', title='Select file', filetypes=(('Text files', '*.txt'), ('all files', '*.*')))
    ruta_img = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=(("Images files", "*.jpg"), ("PNG file", "*.png"), ("TIF files", "*.tif")))
    
    if ruta_img:
        IMG_PATH = ruta_img
        img_show = cv2.imread(ruta_img)
        img_show = cv2.resize(img_show, (200,200))
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        img_pillow = Image.fromarray(img_show)
        img_tk = ImageTk.PhotoImage(img_pillow)

        #si ya hay imagen mostrada, actualizarla, sino crear label
        if panel_img is None:
            panel_img = tk.Label(window, image=img_tk)
            panel_img.image = img_tk
            panel_img.pack() #motrar la imagen
        else:
            panel_img.configure(image=img_tk)
            panel_img.image = img_tk
                
    else:
        print("\n\tERROR AL SELECCIONAR LA IMAGEN\n")
        
    cv2.destroyAllWindows()
    
    showPredLBP()

def loadImgHOG():
    global IMG_PATH, panel_img
  # filedialog.askopenfilename(initialdir='/', title='Select file', filetypes=(('Text files', '*.txt'), ('all files', '*.*')))
    ruta_img = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=(("Images files", "*.jpg"), ("PNG file", "*.png"), ("TIF files", "*.tif")))
    
    if ruta_img:
        IMG_PATH = ruta_img
        img_show = cv2.imread(ruta_img)
        img_show = cv2.resize(img_show, (200,200))
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        img_pillow = Image.fromarray(img_show)
        img_tk = ImageTk.PhotoImage(img_pillow)

        #si ya hay imagen mostrada, actualizarla, sino crear label
        if panel_img is None:
            panel_img = tk.Label(window, image=img_tk)
            panel_img.image = img_tk
            panel_img.pack() #motrar la imagen
        else:
            panel_img.configure(image=img_tk)
            panel_img.image = img_tk
                
    else:
        print("\n\tERROR AL SELECCIONAR LA IMAGEN\n")
        
    cv2.destroyAllWindows()
    
    showPredHOG()
    

def showPredHOG():
    # configuramos el hog
    hog = cv2.HOGDescriptor((128,128), (16,16), (8,8), (8,8), 9)

    # cargamos el modelo
    path_model = os.getcwd()
    # comprobamos si estamos en la carpeta del modelo o no, para poder cargar el modelo

    
    if "Model"  in os.listdir(path_model):
        path_model = os.path.join(path_model, "Model" ,"svm_brain_model_hog.joblib")
    else:
        path_model = os.path.join(path_model, "svm_brain_model_hog.joblib")
    print(f"\n\tactual {path_model}")

    # configuramos clasificador
    clasifier = joblib.load(path_model)

    img_test = cv2.imread(IMG_PATH) # abrimos la imagen seleccionada
    img_test = cv2.resize(img_test, (128,128))
    desc = hog.compute(img_test)

    pred = clasifier.predict(desc.reshape(1,-1))[0]

    print(pred)
    if pred == 1:
        result_text.set("El diagnostico es: Cerebro con tumor (Cancer)")
    elif pred == 0:
        result_text.set("El diagnostico es: Cerebro sano")


def showPredLBP():
    # configuramos el hog

    # cargamos el modelo
    path_model = os.getcwd()
    # comprobamos si estamos en la carpeta del modelo o no, para poder cargar el modelo

    
    if "Model"  in os.listdir(path_model):
        path_model = os.path.join(path_model, "Model" ,"svm_brain_model_lbp.joblib")
    else:
        path_model = os.path.join(path_model, "svm_brain_model_lbp.joblib")
    print(f"\n\tactual {path_model}")

    # configuramos clasificador
    clasifier = joblib.load(path_model)

    img_test = cv2.imread(IMG_PATH) # abrimos la imagen seleccionada
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(img_test, 8, 1, "uniform") # caculamos el lbp
    hist,_ = np.histogram(lbp, 59, (0,59))
    

    pred = clasifier.predict(hist.reshape(1,-1))[0]

    print(pred)
    if pred == 1:
        result_text.set("El diagnostico es: Cerebro con tumor (Cancer)")
    elif pred == 0:
        result_text.set("El diagnostico es: Cerebro sano")

window = tk.Tk() # creamos la ventana
window.title("Ventana de prueba") # ponemos el titulo
window.geometry("400x400") #tamaño de la ventana
result_text = tk.StringVar() # preparamos el texto del resultado obtenido al usar el modelo (cancer o sano)
result_label = tk.Label(window, textvariable=result_text, font=("Arial",14))
result_label.pack(pady=20)
boton_hog = tk.Button(window, text="Adjuntar imagen (modelo con caracteristicas HOG)", command=loadImgHOG)
boton_hog.pack(padx=20)
boton_lbp = tk.Button(window, text="Adjuntar imagen (modelo con caracteristicas LBP)", command=loadImgLBP)
boton_lbp.pack(pady=20)
window.mainloop()