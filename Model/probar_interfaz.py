import tkinter as tk
from tkinter import filedialog
import cv2
from pathlib import Path
import os
import joblib

IMG_PATH = ""

def loadImg():
    global IMG_PATH
  # filedialog.askopenfilename(initialdir='/', title='Select file', filetypes=(('Text files', '*.txt'), ('all files', '*.*')))
    ruta_img = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=(("Images files", "*.jpg"), ("PNG file", "*.png"), ("TIF files", "*.tif")))
    
    if ruta_img:
        img_show = cv2.imread(ruta_img)
        img_show = cv2.resize(img_show, (640,640))
        while True:
            
            cv2.putText(img_show, "Presiona la tecla ESC para cerrar la imagen",  (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("Imagen seleccionada", img_show)
            
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                IMG_PATH = ruta_img
                break
                
    else:
        print("\n\tERROR AL SELECCIONAR LA IMAGEN\n")
        
    cv2.destroyAllWindows()
    
    showPred()
    

def showPred():
    # configuramos el hog
    hog = cv2.HOGDescriptor((128,128), (16,16), (8,8), (8,8), 9)

    # cargamos el modelo
    path_model = os.getcwd()
    # comprobamos si estamos en la carpeta del modelo o no, para poder cargar el modelo
    if os.listdir(path_model)[1] == "Model":
        path_model = os.path.join(path_model, "Model" ,"svm_brain_model.joblib")
    else:
        path_model = os.path.join(path_model, "svm_brain_model.joblib")
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

window = tk.Tk() # creamos la ventana
window.title("Ventana de prueba") # ponemos el titulo
window.geometry("400x400") #tamaño de la ventana
result_text = tk.StringVar() # preparamos el texto del resultado obtenido al usar el modelo (cancer o sano)
result_label = tk.Label(window, textvariable=result_text, font=("Arial",14))
result_label.pack(pady=20)
boton = tk.Button(window, text="Adjuntar imagen", command=loadImg)
boton.pack(padx=20)
window.mainloop()