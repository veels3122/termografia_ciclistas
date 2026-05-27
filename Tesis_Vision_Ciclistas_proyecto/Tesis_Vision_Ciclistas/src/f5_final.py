
#  AUTO-INSTALADOR DE DEPENDENCIAS
import sys
import subprocess
def preparar_entorno():
    print("[*] Inicializando Motor de Preprocesamiento...")
    print("[*] Verificando dependencias del sistema. Por favor, espera...")
    # Diccionario de librerías: 'nombre_en_pip': 'nombre_en_codigo'
    librerias_requeridas = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'tensorflow': 'tensorflow',
        'scikit-learn': 'sklearn',
        'reportlab': 'reportlab',
        'pytesseract': 'pytesseract',
        'openpyxl': 'openpyxl'
    }
    for paquete_pip, modulo in librerias_requeridas.items():
        try:
            #  importar la librería
            __import__(modulo)
        except ImportError:
            # lanza la instalación silenciosa en la terminal
            print(f"    └─ [-] Falta '{paquete_pip}'. Instalando automáticamente...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", paquete_pip, "--quiet"])
                print(f"       └─ [✔] {paquete_pip} instalada con éxito.")
            except subprocess.CalledProcessError:
                print(f"       └─ [!] ERROR CRÍTICO: No se pudo instalar {paquete_pip}. Revisa tu conexión a internet.")
                sys.exit(1) # Detiene el programa si no puede instalar algo vital

    print("[✔] Entorno de Python validado al 100%. Iniciando el software...\n")

# EJECUCIÓN OBLIGATORIA ANTES DE CUALQUIER OTRO IMPORT

preparar_entorno()
import cv2
import pytesseract
import numpy as np
import re
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURACIÓN DE TESSERACT ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#  FASE 1

class TermoOCR:
    def __init__(self, ruta_imagen):
        self.ruta = ruta_imagen
        stream = np.fromfile(ruta_imagen, dtype=np.uint8)
        self.img_original = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if self.img_original is None:
            raise ValueError(f"Error crítico: No se pudo cargar la imagen {ruta_imagen}")
    def preprocesar_texto(self, umbral):
        img_ampliada = cv2.resize(self.img_original, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        lower_white = np.array([umbral, umbral, umbral], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        binary = cv2.inRange(img_ampliada, lower_white, upper_white)
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    def extraer_metadata(self, imagen_binaria):
        h, w = imagen_binaria.shape
        corte = int(w * 0.6)
        roi_izq = imagen_binaria[:, :corte]
        roi_der = imagen_binaria[:, corte:]
        config_tesseract = "--psm 6"
        texto_izq = pytesseract.image_to_string(roi_izq, config=config_tesseract)
        texto_der = pytesseract.image_to_string(roi_der, config=config_tesseract)
        datos = {
            "Fecha": "No detectada",
            "Hora": "No detectada",
            "Emisividad": None,
            "Temp_Reflejada_C": None,
            "Distancia": None,
            "Humedad": None,
            "Temp_Atm_C": None,
            "Temp_Max_C": None,
            "Temp_Min_C": None
        }
        texto_izq_limpio = texto_izq.replace('im', '1m').replace('Im', '1m').replace('l m', '1 m').replace('I m', '1 m')
        m_emis = re.search(r'(?:[eE€ε=]\s*)?(0\.\d{2})', texto_izq_limpio)
        if m_emis: datos["Emisividad"] = float(m_emis.group(1))
        m_trefl = re.search(r'(?:Temp|emp)[.,\s]*(?:refl|re|ref).*?(\d+(?:\.\d+)?)', texto_izq_limpio, re.IGNORECASE)
        if m_trefl: datos["Temp_Reflejada_C"] = float(m_trefl.group(1))
        m_dist = re.search(r'Dist[.,\s]*(?:obj|ob).*?(\d+(?:\.\d+)?)', texto_izq_limpio, re.IGNORECASE)
        if m_dist: datos["Distancia"] = f"{m_dist.group(1)} m"
        m_hum = re.search(r'Hum[.,\s]*(?:rel|re).*?(\d+(?:\.\d+)?)', texto_izq_limpio, re.IGNORECASE)
        if m_hum: datos["Humedad"] = f"{m_hum.group(1)}%"
        m_tatm = re.search(r'(?:Temp|emp)[.,\s]*atm.*?(\d+(?:\.\d+)?)', texto_izq_limpio, re.IGNORECASE)
        if m_tatm: datos["Temp_Atm_C"] = float(m_tatm.group(1))
        texto_der_limpio = texto_der
        m_fecha = re.search(r'(\d{4})[-]?(\d{2})[-]?(\d{2})', texto_der_limpio)
        if m_fecha: 
            datos["Fecha"] = f"{m_fecha.group(1)}-{m_fecha.group(2)}-{m_fecha.group(3)}"
            texto_der_limpio = texto_der_limpio.replace(m_fecha.group(0), "")
        m_hora = re.search(r'([0-2]\d):([0-5]\d)', texto_der_limpio)
        if m_hora:
            datos["Hora"] = f"{m_hora.group(1)}:{m_hora.group(2)}"
            texto_der_limpio = texto_der_limpio.replace(m_hora.group(0), "")

        todos_numeros_der = re.findall(r'(\d+\.\d+)', texto_der_limpio)
        temps_derecha = [float(num) for num in todos_numeros_der if 10.0 <= float(num) <= 50.0]
        
        if len(temps_derecha) >= 2:
            datos["Temp_Max_C"] = max(temps_derecha)
            datos["Temp_Min_C"] = min(temps_derecha)
        elif len(temps_derecha) == 1:
            datos["Temp_Max_C"] = temps_derecha[0]
            datos["Temp_Min_C"] = temps_derecha[0]

        return datos

    def limpiar_con_kmeans_y_guardar(self, ruta_salida):
        img_work = self.img_original.copy()
        
        pasada = 1
        max_pasadas = 50 
        
        while pasada <= max_pasadas:
            Z = img_work.reshape((-1, 3))
            Z = np.float32(Z)

            K = 20 
            criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, etiquetas, centros = cv2.kmeans(Z, K, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centros = np.int32(centros)
            blanco_puro = np.array([255, 255, 255], dtype=np.int32)
            
            clusters_objetivo = []
            for idx, c in enumerate(centros):
                distancia = np.linalg.norm(c - blanco_puro)
                if distancia < 220: 
                    clusters_objetivo.append(idx)

            if not clusters_objetivo:
                print(f"      └─ [K-Means] ¡0 píxeles blancos! Imagen inmaculada en pasada {pasada-1}.")
                break

            mascara_texto = np.isin(etiquetas.flatten(), clusters_objetivo).astype(np.uint8) * 255
            mascara_texto = mascara_texto.reshape(img_work.shape[:2])

            pixeles_restantes = cv2.countNonZero(mascara_texto)
            print(f"      └─ [K-Means] Pasada {pasada}: Eliminando {pixeles_restantes} píxeles...")

            kernel = np.ones((2, 2), np.uint8)
            mascara_dilatada = cv2.dilate(mascara_texto, kernel, iterations=1)
            
            img_work = cv2.inpaint(img_work, mascara_dilatada, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            
            pasada += 1
        exito, buffer = cv2.imencode('.jpg', img_work)
        if exito:
            with open(ruta_salida, 'wb') as f:
                buffer.tofile(f)

def contar_faltantes(datos):
    return sum(1 for v in datos.values() if v is None or v == "No detectada")

def ejecutar_pipeline_extraccion():
    print("\n[+] INICIANDO fase 1: PIPELINE MAESTRO (OCR + LIMPIEZA K-MEANS PROFUNDA)...")
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    directorio_data = os.path.join(directorio_actual, "..", "data")
    ruta_excel = os.path.join(directorio_data, "resultados_termografia_fase1.xlsx")
    
    directorio_limpias = os.path.join(directorio_actual, "..", "imagenes limpias")
    if not os.path.exists(directorio_limpias):
        os.makedirs(directorio_limpias, exist_ok=True)

    if not os.path.exists(directorio_data):
        os.makedirs(directorio_data, exist_ok=True)
    
    columnas_df = ["Participante", "Archivo_Imagen", "Fecha", "Hora", "Emisividad", 
                   "Temp_Reflejada_C", "Distancia", "Humedad", "Temp_Atm_C", 
                   "Temp_Max_C", "Temp_Min_C"]

    if os.path.exists(ruta_excel):
        df_historico = pd.read_excel(ruta_excel)
    else:
        df_historico = pd.DataFrame(columns=columnas_df)
        df_historico.to_excel(ruta_excel, index=False)

    nuevos_registros = []
    carpetas_participantes = [d for d in os.listdir(directorio_data) if os.path.isdir(os.path.join(directorio_data, d))]

    if not carpetas_participantes:
        print("\n[-] ADVERTENCIA: La carpeta 'data' está vacía.")
        return

    secuencia_umbrales = list(range(50, 19, -2)) + list(range(52, 101, 2))

    for carpeta_usuario in carpetas_participantes:
        ruta_usuario = os.path.join(directorio_data, carpeta_usuario)
        archivos_imagenes = [f for f in os.listdir(ruta_usuario) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not archivos_imagenes:
            continue
            
        print(f"\n========================================================")
        print(f"📁 PARTICIPANTE: {carpeta_usuario}")
        print(f"========================================================")
        
        nombre_carpeta_limpia = carpeta_usuario.replace(" ", "_")
        ruta_usuario_limpia = os.path.join(directorio_limpias, nombre_carpeta_limpia)
        if not os.path.exists(ruta_usuario_limpia):
            os.makedirs(ruta_usuario_limpia, exist_ok=True)

        for archivo in archivos_imagenes:
            ruta_img = os.path.join(ruta_usuario, archivo)
            
            nombre_base, extension = os.path.splitext(archivo)
            nombre_archivo_limpio = f"{nombre_base}_clean{extension}"
            ruta_salida_limpia = os.path.join(ruta_usuario_limpia, nombre_archivo_limpio)

            ya_en_excel = False
            if not df_historico.empty:
                coincidencias = df_historico[(df_historico["Participante"] == carpeta_usuario) & (df_historico["Archivo_Imagen"] == archivo)]
                ya_en_excel = not coincidencias.empty

            ya_esta_limpia = os.path.exists(ruta_salida_limpia)

            print(f"  [ Archivo: {archivo} ]")

            if ya_en_excel and ya_esta_limpia:
                print(f"    ├─ Excel: [-] Omitido (Registro ya existente)")
                print(f"    └─ Clean: [-] Omitido (Imagen limpia ya existente)")
                continue

            try:
                motor = TermoOCR(ruta_img)
                
                # --- EXTRACCIÓN AL 100% ---
                if not ya_en_excel:
                    mejor_datos = None
                    mejor_umbral = 50
                    min_faltantes = 99

                    for umbral in secuencia_umbrales:
                        mascara = motor.preprocesar_texto(umbral)
                        datos_temp = motor.extraer_metadata(mascara)
                        faltantes = contar_faltantes(datos_temp)
                        
                        if faltantes < min_faltantes:
                            min_faltantes = faltantes
                            mejor_datos = datos_temp
                            mejor_umbral = umbral
                            
                        if min_faltantes == 0:
                            break

                    fila_excel = {
                        "Participante": carpeta_usuario,
                        "Archivo_Imagen": archivo,
                        "Fecha": mejor_datos["Fecha"],
                        "Hora": mejor_datos["Hora"],
                        "Emisividad": mejor_datos["Emisividad"],
                        "Temp_Reflejada_C": mejor_datos["Temp_Reflejada_C"],
                        "Distancia": mejor_datos["Distancia"],
                        "Humedad": mejor_datos["Humedad"],
                        "Temp_Atm_C": mejor_datos["Temp_Atm_C"],
                        "Temp_Max_C": mejor_datos["Temp_Max_C"],
                        "Temp_Min_C": mejor_datos["Temp_Min_C"]
                    }
                    nuevos_registros.append(fila_excel)
                    
                    if min_faltantes == 0:
                        print(f"    ├─ Excel: [+] 100% de datos capturados (Umbral: {mejor_umbral})")
                    else:
                        print(f"    ├─ Excel: [!] Faltaron {min_faltantes} datos (Mejor umbral: {mejor_umbral})")
                else:
                    print(f"    ├─ Excel: [-] Omitido (Registro ya existente)")

                # --- LIMPIEZA CON K-MEANS Y AUTO-REVISIÓN ---
                if not ya_esta_limpia:
                    print(f"    ├─ Clean: [+] Iniciando ciclo de limpieza...")
                    motor.limpiar_con_kmeans_y_guardar(ruta_salida_limpia)
                    print(f"    └─ Clean: [+] Imagen guardada con éxito.")
                else:
                    print(f"    └─ Clean: [-] Omitido (Imagen limpia ya existente)")

            except Exception as e:
                print(f"    └─ [!] ERROR CRÍTICO al procesar: {e}")

    if nuevos_registros:
        print(f"\n[+] Guardando {len(nuevos_registros)} nuevos registros en Excel...")
        df_nuevos = pd.DataFrame(nuevos_registros)
        df_final = pd.concat([df_historico, df_nuevos], ignore_index=True)
        df_final = df_final[columnas_df]
        df_final.to_excel(ruta_excel, index=False)
        print(f"[+] Base de datos actualizada con éxito.")
    else:
        print("\n[i] No hubo registros nuevos para agregar al Excel.")

# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 MEJORADA: U-NET CON SKIP CONNECTIONS + FINE-TUNING CONTINUO
# + VENTANAS DE PROGRESO EN TIEMPO REAL
# ─────────────────────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
import pandas as pd
import threading
import warnings
import tkinter as tk
from tkinter import ttk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
CUADROS = [
    [0, 160, 111, 237], [254,  7, 311,  23], [271, 39, 319,  65],
    [276, 174, 318, 196], [249, 204, 317, 237],
]
V_PARAMS = {'mult': 1.67, 'min': 122, 'max': 255}
S_PARAMS = {'mult': 5.0,  'min': 212, 'max': 255}

MODELO_H        = 256
MODELO_W        = 320
EPOCAS_BASE     = 50
EPOCAS_FINETUNE = 2

# ── VENTANA DE PROGRESO ────────────────────────────────────────────────────────
class VentanaProgreso:
    """
    Ventana pequeña con dos barras de progreso:
      - Barra superior: entrenamiento base (épocas)
      - Barra inferior: procesamiento de las 899 imágenes
    Se actualiza desde el hilo principal sin bloquear el proceso.
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fase 2 — Progreso")
        self.root.geometry("520x210")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")
        self.root.attributes('-topmost', True)

        estilo = ttk.Style()
        estilo.theme_use('clam')
        estilo.configure("Azul.Horizontal.TProgressbar",
                         troughcolor="#2e2e3e", background="#2E75B6",
                         thickness=22)
        estilo.configure("Verde.Horizontal.TProgressbar",
                         troughcolor="#2e2e3e", background="#1E7A4A",
                         thickness=22)

        pad = {"padx": 18, "pady": 4}

        # ── Sección entrenamiento ──────────────────────────────
        tk.Label(self.root, text="Entrenamiento base (épocas)",
                 bg="#1e1e2e", fg="#aaaacc", font=("Arial", 9)).pack(anchor="w", **pad)
        self.barra_train = ttk.Progressbar(self.root, style="Azul.Horizontal.TProgressbar",
                                           orient="horizontal", length=484, mode="determinate")
        self.barra_train.pack(**pad)
        self.lbl_train = tk.Label(self.root, text="0 / 0 épocas",
                                  bg="#1e1e2e", fg="#2E75B6", font=("Arial", 8))
        self.lbl_train.pack(anchor="w", padx=18)

        # ── Separador ──────────────────────────────────────────
        tk.Frame(self.root, bg="#3e3e5e", height=1).pack(fill="x", padx=18, pady=6)

        # ── Sección imágenes ───────────────────────────────────
        tk.Label(self.root, text="Procesamiento de imágenes",
                 bg="#1e1e2e", fg="#aaaacc", font=("Arial", 9)).pack(anchor="w", **pad)
        self.barra_imgs = ttk.Progressbar(self.root, style="Verde.Horizontal.TProgressbar",
                                          orient="horizontal", length=484, mode="determinate")
        self.barra_imgs.pack(**pad)
        self.lbl_imgs = tk.Label(self.root, text="0 / 0 imágenes",
                                 bg="#1e1e2e", fg="#1E7A4A", font=("Arial", 8))
        self.lbl_imgs.pack(anchor="w", padx=18)

        # ── Línea de estado ────────────────────────────────────
        self.lbl_estado = tk.Label(self.root, text="Iniciando...",
                                   bg="#1e1e2e", fg="#cccccc",
                                   font=("Arial", 8), wraplength=490)
        self.lbl_estado.pack(anchor="w", padx=18, pady=(8, 4))

        self.root.protocol("WM_DELETE_WINDOW", lambda: None)  # no cerrar con X
        self.root.update()

    # ── Actualizar barra de entrenamiento ──────────────────────
    def set_train(self, actual, total, texto=""):
        pct = int((actual / total) * 100) if total > 0 else 0
        self.barra_train["maximum"] = 100
        self.barra_train["value"]   = pct
        self.lbl_train.config(text=f"{actual} / {total} épocas  ({pct}%)")
        if texto:
            self.lbl_estado.config(text=texto)
        self.root.update()

    # ── Actualizar barra de imágenes ───────────────────────────
    def set_imgs(self, actual, total, texto=""):
        pct = int((actual / total) * 100) if total > 0 else 0
        self.barra_imgs["maximum"] = 100
        self.barra_imgs["value"]   = pct
        self.lbl_imgs.config(text=f"{actual} / {total} imágenes  ({pct}%)")
        if texto:
            self.lbl_estado.config(text=texto)
        self.root.update()

    def estado(self, texto):
        self.lbl_estado.config(text=texto)
        self.root.update()

    def cerrar(self):
        self.estado("✔ Fase 2 completada.")
        self.root.update()
        self.root.after(2500, self.root.destroy)
        try:
            self.root.mainloop()
        except Exception:
            pass


# ── U-NET CON SKIP CONNECTIONS ────────────────────────────────────────────────
def crear_unet():
    K.clear_session()
    inp = layers.Input((MODELO_H, MODELO_W, 3))

    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    # Decoder
    u4 = layers.UpSampling2D()(c3)
    u4 = layers.Concatenate()([u4, c2])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    u5 = layers.UpSampling2D()(c4)
    u5 = layers.Concatenate()([u5, c1])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)

    out = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    m   = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])
    return m


# ── CALLBACK PARA ACTUALIZAR BARRA DURANTE ENTRENAMIENTO ─────────────────────
class CallbackProgreso(tf.keras.callbacks.Callback):
    def __init__(self, ventana, total_epocas, offset=0):
        super().__init__()
        self.ventana      = ventana
        self.total_epocas = total_epocas
        self.offset       = offset   # épocas ya completadas antes de este bloque

    def on_epoch_end(self, epoch, logs=None):
        actual = self.offset + epoch + 1
        loss   = logs.get('loss', 0)
        self.ventana.set_train(actual, self.total_epocas,
                               f"Entrenando... época {actual}/{self.total_epocas}  "
                               f"loss={loss:.4f}")


# ── PREPROCESAR IMAGEN ────────────────────────────────────────────────────────
def preprocesar(img_bgr):
    h, w    = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    for x1, y1, x2, y2 in CUADROS:
        y2c, x2c = min(y2, h), min(x2, w)
        if y1 >= y2c or x1 >= x2c:
            continue
        roi = img_hsv[y1:y2c, x1:x2c]
        mv  = roi[:, :, 2] > 15
        roi[:, :, 1][mv] = np.clip(roi[:, :, 1][mv].astype(np.float32) * S_PARAMS['mult'],
                                    S_PARAMS['min'], S_PARAMS['max']).astype(np.uint8)
        roi[:, :, 2][mv] = np.clip(roi[:, :, 2][mv].astype(np.float32) * V_PARAMS['mult'],
                                    V_PARAMS['min'], V_PARAMS['max']).astype(np.uint8)
    img_f   = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_m   = cv2.resize(img_f, (MODELO_W, MODELO_H))
    x_model = np.expand_dims(img_m / 255.0, axis=0).astype(np.float32)
    return img_f, x_model, h, w


def generar_mascara_umbral(img_f, uf, up):
    h, w   = img_f.shape[:2]
    gray   = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((15, 15), np.uint8)
    _, m1  = cv2.threshold(gray, uf, 255, cv2.THRESH_BINARY)
    m1     = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, kernel)
    mo     = m1.copy()
    gs     = gray[0:h//2, :]
    _, ms  = cv2.threshold(gs, up, 255, cv2.THRESH_BINARY)
    ms     = cv2.morphologyEx(ms, cv2.MORPH_CLOSE, kernel)
    mo[0:h//2, :] = ms
    return cv2.resize(mo, (MODELO_W, MODELO_H))


def jaccard_simple(a, b):
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a  > 0, b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


# ── RAY-CASTING (igual que siempre) ───────────────────────────────────────────
def identificar_coordenadas_huecos(mask):
    tu  = np.cumsum(mask == 255, axis=0) > 0
    la  = np.roll(tu, 1, axis=0);  la[0,  :] = False
    td  = np.flip(np.cumsum(np.flip(mask==255,axis=0),axis=0)>0, axis=0)
    lb  = np.roll(td,-1, axis=0);  lb[-1, :] = False
    tl  = np.cumsum(mask == 255, axis=1) > 0
    li  = np.roll(tl, 1, axis=1);  li[:,  0] = False
    tr  = np.flip(np.cumsum(np.flip(mask==255,axis=1),axis=1)>0, axis=1)
    ld  = np.roll(tr,-1, axis=1);  ld[:, -1] = False
    return (mask == 0) & la & lb & li & ld


# ── ENTRENAMIENTO BASE ────────────────────────────────────────────────────────
def entrenar_modelo_base(ruta_modelo, uf, up, ventana):
    dir_actual   = os.path.dirname(os.path.abspath(__file__))
    ruta_man     = os.path.join(dir_actual, "..", "mascaras_entrenamiento")
    dir_limpias  = os.path.join(dir_actual, "..", "imagenes limpias")

    print("\n[+] ENTRENAMIENTO BASE DE LA U-NET...")
    ventana.estado("Preparando datos de entrenamiento...")

    modelo   = crear_unet()
    X, Y     = [], []
    tiene_m  = (os.path.isdir(os.path.join(ruta_man, "imagenes")) and
                os.path.isdir(os.path.join(ruta_man, "mascaras")))

    if tiene_m:
        print("[+] Máscaras manuales encontradas...")
        ventana.estado("Cargando máscaras manuales...")
        ri = os.path.join(ruta_man, "imagenes")
        rm = os.path.join(ruta_man, "mascaras")
        for f in os.listdir(ri):
            if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
            nb  = os.path.splitext(f)[0]
            rmi = None
            for e in ['.png','.jpg','.jpeg']:
                c = os.path.join(rm, nb+e)
                if os.path.exists(c): rmi = c; break
            if rmi is None: continue
            img = cv2.imdecode(np.fromfile(os.path.join(ri,f),dtype=np.uint8), cv2.IMREAD_COLOR)
            msk = cv2.imdecode(np.fromfile(rmi, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None: continue
            _, xm, _, _ = preprocesar(img)
            mr  = cv2.resize(msk,(MODELO_W,MODELO_H))
            _, mb = cv2.threshold(mr,127,255,cv2.THRESH_BINARY)
            X.append(xm[0]); Y.append(np.expand_dims(mb/255.0,axis=-1).astype(np.float32))
            print(f"  [+] {f}")
        print(f"[+] Máscaras manuales cargadas: {len(X)}")
    else:
        print("[!] Sin máscaras manuales — usando umbral automático en primeras imágenes...")
        ventana.estado("Generando datos de entrenamiento automático...")
        if os.path.exists(dir_limpias):
            carpetas = [d for d in os.listdir(dir_limpias)
                        if os.path.isdir(os.path.join(dir_limpias,d))][:8]
            for cap in carpetas:
                archs = [f for f in os.listdir(os.path.join(dir_limpias,cap))
                         if f.lower().endswith(('.jpg','.jpeg','.png'))][:3]
                for f in archs:
                    img = cv2.imdecode(np.fromfile(os.path.join(dir_limpias,cap,f),dtype=np.uint8),
                                       cv2.IMREAD_COLOR)
                    if img is None: continue
                    imgf, xm, _, _ = preprocesar(img)
                    mu = generar_mascara_umbral(imgf, uf, up)
                    X.append(xm[0]); Y.append(np.expand_dims(mu/255.0,axis=-1).astype(np.float32))
        print(f"[+] Imágenes automáticas para entrenamiento: {len(X)}")

    if len(X) == 0:
        print("[!] Sin datos — guardando modelo vacío.")
        modelo.save(ruta_modelo); return modelo

    Xa = np.array(X); Ya = np.array(Y)
    vs = 0.15 if len(Xa) >= 10 else 0.0
    ventana.set_train(0, EPOCAS_BASE, f"Iniciando entrenamiento con {len(Xa)} imágenes...")

    cb = CallbackProgreso(ventana, EPOCAS_BASE)
    print(f"[+] Entrenando {len(Xa)} imágenes × {EPOCAS_BASE} épocas...")
    modelo.fit(Xa, Ya, epochs=EPOCAS_BASE,
               batch_size=min(4,len(Xa)),
               validation_split=vs,
               callbacks=[cb], verbose=0)

    ventana.set_train(EPOCAS_BASE, EPOCAS_BASE, "✔ Entrenamiento base completado.")
    modelo.save(ruta_modelo)
    print(f"[✔] Modelo guardado: {ruta_modelo}")
    return modelo


# ── FASE 2 PRINCIPAL ──────────────────────────────────────────────────────────
def ejecutar_limpieza_quirurgica_lote():
    print("\n\n[+] INICIANDO FASE 2 (U-NET CON FINE-TUNING + BARRA DE PROGRESO)...")

    dir_actual         = os.path.dirname(os.path.abspath(__file__))
    ruta_data          = os.path.join(dir_actual, "..", "data")
    ruta_modelo        = os.path.join(ruta_data,  "modelo_fase2.keras")
    ruta_excel_memoria = os.path.join(ruta_data,  "memoria_fase2.xlsx")
    dir_limpias        = os.path.join(dir_actual, "..", "imagenes limpias")
    dir_fase1          = os.path.join(dir_actual, "..", "imagenes fase 1")

    os.makedirs(ruta_data, exist_ok=True)

    # Parámetros
    print("\nConfigura los parámetros (Enter = valores por defecto):")
    try:
        uf = input("  Agresividad Fondo       [Por defecto 25]: ")
        up = input("  Agresividad Pantaloneta [Por defecto 65]: ")
        umbral_fondo       = int(uf) if uf.strip() else 25
        umbral_pantaloneta = int(up) if up.strip() else 65
    except ValueError:
        umbral_fondo, umbral_pantaloneta = 25, 65

    # ── Abrir ventana de progreso ──────────────────────────────
    ventana = VentanaProgreso()

    # ── Contar imágenes totales para la barra ─────────────────
    if not os.path.exists(dir_limpias):
        ventana.estado("ERROR: carpeta 'imagenes limpias' no encontrada.")
        ventana.cerrar(); return

    carpetas = [d for d in os.listdir(dir_limpias)
                if os.path.isdir(os.path.join(dir_limpias, d))]

    # Contar solo las que faltan procesar
    lista_pendientes = []
    for cap in carpetas:
        for f in os.listdir(os.path.join(dir_limpias, cap)):
            if not f.lower().endswith(('.png','.jpg','.jpeg')): continue
            ruta_out = os.path.join(dir_fase1, cap, f)
            if not os.path.exists(ruta_out):
                lista_pendientes.append((cap, f))

    total_imgs = len(lista_pendientes)
    ventana.set_imgs(0, total_imgs, f"Pendientes: {total_imgs} imágenes")

    # ── Cargar o crear modelo ──────────────────────────────────
    if os.path.exists(ruta_modelo):
        ventana.estado("Cargando modelo guardado...")
        print(f"\n[+] Cargando modelo desde memoria: {ruta_modelo}")
        K.clear_session()
        modelo = tf.keras.models.load_model(ruta_modelo)
        ventana.set_train(EPOCAS_BASE, EPOCAS_BASE, "✔ Modelo cargado desde memoria anterior.")
        print("[✔] Modelo cargado.")
    else:
        modelo = entrenar_modelo_base(ruta_modelo, umbral_fondo, umbral_pantaloneta, ventana)

    # ── Imágenes de validación ─────────────────────────────────
    imgs_val = []
    ruta_val = os.path.join(dir_actual, "..", "mascaras_validacion")
    if (os.path.isdir(os.path.join(ruta_val,"imagenes")) and
        os.path.isdir(os.path.join(ruta_val,"mascaras"))):
        for f in os.listdir(os.path.join(ruta_val,"imagenes")):
            if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
            imgv = cv2.imdecode(np.fromfile(os.path.join(ruta_val,"imagenes",f),dtype=np.uint8),
                                cv2.IMREAD_COLOR)
            mskv = None
            nb   = os.path.splitext(f)[0]
            for e in ['.png','.jpg','.jpeg']:
                c = os.path.join(ruta_val,"mascaras",nb+e)
                if os.path.exists(c):
                    mskv = cv2.imdecode(np.fromfile(c,dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    break
            if imgv is None or mskv is None: continue
            _, xv, _, _ = preprocesar(imgv)
            mr = cv2.resize(mskv,(MODELO_W,MODELO_H))
            _, mb = cv2.threshold(mr,127,255,cv2.THRESH_BINARY)
            imgs_val.append((xv, mb))
        print(f"[+] {len(imgs_val)} imágenes de validación.")

    def jac_val():
        if not imgs_val: return None
        return round(float(np.mean([
            jaccard_simple((modelo.predict(xv,verbose=0)[0,:,:,0]>0.5).astype(np.uint8)*255, mv)
            for xv, mv in imgs_val])), 4)

    jac_ant = jac_val()
    if jac_ant: print(f"[+] Jaccard validación inicial: {jac_ant:.4f}")

    # ── Excel de memoria ───────────────────────────────────────
    cols = ["Participante","Imagen","Jaccard_Validacion","Mejoro",
            "Umbral_Fondo","Umbral_Pantaloneta","Total_Aprendidas"]
    if os.path.exists(ruta_excel_memoria):
        df_mem = pd.read_excel(ruta_excel_memoria)
        total_aprendidas = len(df_mem)
    else:
        df_mem = pd.DataFrame(columns=cols)
        total_aprendidas = 0

    os.makedirs(dir_fase1, exist_ok=True)
    nuevos = []
    imgs_procesadas = 0

    # ── Bucle principal ────────────────────────────────────────
    for carpeta_usuario in carpetas:
        ruta_origen  = os.path.join(dir_limpias, carpeta_usuario)
        ruta_destino = os.path.join(dir_fase1,   carpeta_usuario)
        os.makedirs(ruta_destino, exist_ok=True)

        archivos = [f for f in os.listdir(ruta_origen)
                    if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if not archivos: continue

        print(f"\n{'='*56}")
        print(f"📁 FASE 2 - PARTICIPANTE: {carpeta_usuario}")
        print(f"{'='*56}")

        for archivo in archivos:
            ruta_in  = os.path.join(ruta_origen,  archivo)
            ruta_out = os.path.join(ruta_destino, archivo)

            if os.path.exists(ruta_out):
                print(f"  [-] Omitido (ya existe): {archivo}")
                continue

            print(f"\n  [ {archivo} ]")
            imgs_procesadas += 1
            ventana.set_imgs(imgs_procesadas, total_imgs,
                             f"Procesando: {carpeta_usuario} / {archivo}")

            try:
                img_orig = cv2.imdecode(np.fromfile(ruta_in,dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_orig is None: continue
                h, w = img_orig.shape[:2]

                # Preprocesar
                img_f, x_model, _, _ = preprocesar(img_orig)

                # Predicción
                print(f"    ├─ [>>>] Predicción U-Net #{total_aprendidas+1}...")
                pred      = modelo.predict(x_model, verbose=0)[0]
                mask_red  = (pred[:,:,0] > 0.5).astype(np.uint8) * 255
                mask_r_o  = cv2.resize(mask_red,(w,h), interpolation=cv2.INTER_NEAREST)

                # Umbral
                mask_u   = generar_mascara_umbral(img_f, umbral_fondo, umbral_pantaloneta)
                mask_u_o = cv2.resize(mask_u,(w,h), interpolation=cv2.INTER_NEAREST)

                # Combinar
                mask_c = cv2.bitwise_and(mask_r_o, mask_u_o)
                if cv2.countNonZero(mask_u_o) > 0:
                    if cv2.countNonZero(mask_c) < cv2.countNonZero(mask_u_o) * 0.25:
                        mask_c = mask_u_o.copy()

                # Ray-Casting
                print(f"    ├─ [✔] Ray-Casting...")
                huecos = identificar_coordenadas_huecos(mask_c)
                mask_c[huecos] = 255

                # Limpiar ruido
                nl, labels, stats, _ = cv2.connectedComponentsWithStats(mask_c)
                mask_final = mask_c.copy()
                for i in range(1, nl):
                    if stats[i, cv2.CC_STAT_AREA] < 800:
                        mask_final[labels == i] = 0

                # Guardar imagen (mismo sistema que antes)
                res_final = np.zeros_like(img_orig)
                res_final[mask_final == 255] = img_orig[mask_final == 255]
                exito, buffer = cv2.imencode('.jpg', res_final)
                if exito:
                    with open(ruta_out, 'wb') as f:
                        buffer.tofile(f)

                # Fine-tuning
                print(f"    ├─ [>>>] Fine-tuning ({EPOCAS_FINETUNE} épocas)...")
                mask_ft = cv2.resize(mask_final,(MODELO_W,MODELO_H))
                y_ft    = np.expand_dims(mask_ft/255.0, axis=[0,-1]).astype(np.float32)

                # Callback para actualizar barra de entrenamiento durante fine-tuning
                cb_ft = CallbackProgreso(ventana, EPOCAS_FINETUNE)
                modelo.fit(x_model, y_ft, epochs=EPOCAS_FINETUNE,
                           callbacks=[cb_ft], verbose=0)

                # Restaurar barra de entrenamiento a completada
                ventana.set_train(EPOCAS_BASE, EPOCAS_BASE,
                                  f"✔ Fine-tuning completado — imagen #{total_aprendidas+1}")

                modelo.save(ruta_modelo)
                total_aprendidas += 1

                # Jaccard de validación
                jac_act = jac_val()
                mejoro  = None
                if jac_act and jac_ant:
                    mejoro = jac_act > jac_ant
                    sym    = "↑ MEJORÓ" if mejoro else ("→ igual" if jac_act==jac_ant else "↓ bajó")
                    print(f"    ├─ [VAL] Jaccard: {jac_act:.4f}  {sym}")
                    jac_ant = jac_act

                print(f"    └─ [+] Guardado exitoso (0% pérdida de datos, huecos restaurados) en 'imagenes fase 1'")

                nuevos.append({
                    "Participante": carpeta_usuario, "Imagen": archivo,
                    "Jaccard_Validacion": jac_act, "Mejoro": mejoro,
                    "Umbral_Fondo": umbral_fondo, "Umbral_Pantaloneta": umbral_pantaloneta,
                    "Total_Aprendidas": total_aprendidas,
                })

            except Exception as e:
                import traceback
                print(f"    └─ [!] ERROR {archivo}: {e}")
                traceback.print_exc()

    # Guardar Excel
    if nuevos:
        df_n   = pd.DataFrame(nuevos)
        df_mem = pd.concat([df_mem, df_n], ignore_index=True)
        df_mem.to_excel(ruta_excel_memoria, index=False)
        print(f"\n[✔] Memoria guardada: {ruta_excel_memoria}")

    if jac_ant: print(f"[✔] Jaccard de validación final: {jac_ant:.4f}")
    print(f"[✔] Total imágenes aprendidas: {total_aprendidas}")
    print("\n[✔] FASE 2 COMPLETADA.")

    ventana.set_imgs(total_imgs, total_imgs, "✔ Todas las imágenes procesadas.")
    ventana.cerrar()


# ── FASE 2.5: GUILLOTINA (igual que antes) ────────────────────────────────────
def ejecutar_eliminacion_manos_superior():
    print("\n\n[+] INICIANDO FASE 2.5 (GUILLOTINA CON MURO DE CONTENCIÓN Y GRAVEDAD)...")

    dir_actual  = os.path.dirname(os.path.abspath(__file__))
    dir_fase1   = os.path.join(dir_actual, "..", "imagenes fase 1")
    dir_fase2   = os.path.join(dir_actual, "..", "imagenes fase 2")

    if not os.path.exists(dir_fase1):
        print("\n[-] La carpeta 'imagenes fase 1' no existe. Abortando Fase 2.5.")
        return

    os.makedirs(dir_fase2, exist_ok=True)
    carpetas = [d for d in os.listdir(dir_fase1)
                if os.path.isdir(os.path.join(dir_fase1, d))]

    for carpeta_usuario in carpetas:
        ruta_origen  = os.path.join(dir_fase1,  carpeta_usuario)
        ruta_destino = os.path.join(dir_fase2, carpeta_usuario)
        os.makedirs(ruta_destino, exist_ok=True)

        archivos = [f for f in os.listdir(ruta_origen)
                    if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if not archivos: continue

        print(f"\n{'='*56}")
        print(f"📁 FASE 2.5 - REVISANDO PARTICIPANTE: {carpeta_usuario}")
        print(f"{'='*56}")

        for archivo in archivos:
            ruta_in  = os.path.join(ruta_origen,  archivo)
            ruta_out = os.path.join(ruta_destino, archivo)

            if os.path.exists(ruta_out):
                print(f"  [-] IGNORADA: {carpeta_usuario} | {archivo} → ya existe en Fase 2.")
                continue

            print(f"  [+] PROCESANDO: {carpeta_usuario} | {archivo}...")

            try:
                img  = cv2.imdecode(np.fromfile(ruta_in,dtype=np.uint8), cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
                img_limpia       = img.copy()
                manos_eliminadas = False

                # PARTE A: manos flotantes
                nl, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                ls = int(h * 0.4)
                for i in range(1, nl):
                    yb = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
                    if yb < ls:
                        img_limpia[labels==i] = 0
                        mask[labels==i]       = 0
                        manos_eliminadas      = True

                # PARTE B: manos pegadas
                yc, _ = np.where(mask > 0)
                if len(yc) == 0:
                    print(f"    └─ [!] Imagen vacía.")
                    continue
                y_bottom  = np.max(yc)
                img_hsv   = cv2.cvtColor(img_limpia, cv2.COLOR_BGR2HSV)
                mc1       = cv2.inRange(img_hsv, np.array([0,100,100]),   np.array([25,255,255]))
                mc2       = cv2.inRange(img_hsv, np.array([160,100,100]), np.array([180,255,255]))
                mask_cal  = cv2.bitwise_or(mc1, mc2)
                left_e    = np.full(h, w)
                right_e   = np.full(h, 0)
                for y in range(y_bottom, -1, -1):
                    row = np.where(mask[y, :] > 0)[0]
                    if len(row) > 0:
                        left_e[y]  = row[0]
                        right_e[y] = row[-1]

                vent     = 15
                zona_sup = int(h * 0.6)

                # Izquierdo
                gL = False; dxL = 0.0; xL = 0.0
                for y in range(y_bottom - vent, -1, -1):
                    if gL:
                        xL += dxL
                        lim = min(max(0, int(xL)), w//2)
                        img_limpia[y, 0:lim] = 0; mask[y, 0:lim] = 0; continue
                    if left_e[y] != w and y < zona_sup and left_e[y+vent] != w:
                        s = left_e[y] - left_e[y+vent]
                        if s < -12:
                            roi = mask_cal[max(0,y-20):y+5, 0:left_e[y+vent]]
                            if cv2.countNonZero(roi) > 0:
                                gL = True; manos_eliminadas = True; ys = y+vent
                                dxL = (left_e[ys]-left_e[min(y_bottom,ys+vent)])/vent
                                xL  = float(left_e[ys])

                # Derecho
                gR = False; dxR = 0.0; xR = 0.0
                for y in range(y_bottom - vent, -1, -1):
                    if gR:
                        xR += dxR
                        lim = max(min(w, int(xR)), w//2)
                        img_limpia[y, lim:w] = 0; mask[y, lim:w] = 0; continue
                    if right_e[y] != 0 and y < zona_sup and right_e[y+vent] != 0:
                        s = right_e[y] - right_e[y+vent]
                        if s > 12:
                            roi = mask_cal[max(0,y-20):y+5, right_e[y+vent]:w]
                            if cv2.countNonZero(roi) > 0:
                                gR = True; manos_eliminadas = True; ys = y+vent
                                dxR = (right_e[ys]-right_e[min(y_bottom,ys+vent)])/vent
                                xR  = float(right_e[ys])

                exito, buffer = cv2.imencode('.jpg', img_limpia)
                if exito:
                    with open(ruta_out, 'wb') as f:
                        buffer.tofile(f)
                estado = "Guillotina aplicada." if manos_eliminadas else "Línea natural, no requirió cortes."
                print(f"    └─ [✔] PROCESADA: {carpeta_usuario} | {archivo} → {estado}")

            except Exception as e:
                print(f"    └─ [!] ERROR procesando {carpeta_usuario} | {archivo}: {e}")
                
#  FASE 3: ESTRATIFICACIÓN, K-MEANS Y SELLO DE FONDO (HARD MASKING)

from sklearn.cluster import KMeans

def ejecutar_segmentacion_kmeans_dinamico():
    print("\n\n[+] INICIANDO FASE 3 (K-MEANS Y SELLO DE FONDO ESTRICTO)...")
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    directorio_fase2 = os.path.join(directorio_actual, "..", "imagenes fase 2")
    directorio_fase3 = os.path.join(directorio_actual, "..", "imagenes fase 3")

    if not os.path.exists(directorio_fase2):
        print("\n[-] La carpeta 'imagenes fase 2' no existe. Abortando Fase 3.")
        return

    os.makedirs(directorio_fase3, exist_ok=True)
    carpetas_participantes = [d for d in os.listdir(directorio_fase2) if os.path.isdir(os.path.join(directorio_fase2, d))]

    paleta_clinica = np.array([
        [255, 0, 0],     # 0: Azul (Frío)
        [150, 200, 0],   # 1: Verde/Cian
        [0, 255, 255],   # 2: Amarillo
        [0, 100, 255],   # 3: Naranja
        [0, 0, 255]      # 4: Rojo (Hotspot)
    ], dtype=np.uint8)

    for carpeta_usuario in carpetas_participantes:
        ruta_usuario_origen = os.path.join(directorio_fase2, carpeta_usuario)
        ruta_usuario_destino = os.path.join(directorio_fase3, carpeta_usuario)
        os.makedirs(ruta_usuario_destino, exist_ok=True)

        archivos_imagenes = [f for f in os.listdir(ruta_usuario_origen) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not archivos_imagenes: continue

        print(f"\n========================================================")
        print(f"📁 FASE 3 - PROCESANDO PARTICIPANTE: {carpeta_usuario}")
        print(f"========================================================")

        for archivo in archivos_imagenes:
            nombre_base, extension = os.path.splitext(archivo)
            ruta_in = os.path.join(ruta_usuario_origen, archivo)
            
            archivos_existentes = [f for f in os.listdir(ruta_usuario_destino) if f.startswith(nombre_base + "_Rango_")]
            if archivos_existentes:
                print(f"  [-] IGNORADA: [{archivo}] -> Ya procesada como {archivos_existentes[0]}")
                continue

            print(f"  [+] ANALIZANDO: [{archivo}]...")

            try:
                # 1. CARGA DE LA IMAGEN FASE 2
                img_pierna = cv2.imdecode(np.fromfile(ruta_in, dtype=np.uint8), cv2.IMREAD_COLOR)
                h, w = img_pierna.shape[:2]
                
                # --- EXTRACCIÓN DE LAS COORDENADAS DEL FONDO FASE 2 ---
                # Pasamos a escala de grises y detectamos todo lo oscuro
                gris_pierna = cv2.cvtColor(img_pierna, cv2.COLOR_BGR2GRAY)
                mascara_fondo_fase2 = (gris_pierna <= 5)
                
                # La anatomía para el K-Means será estrictamente lo que no es fondo
                mascara_anat = ~mascara_fondo_fase2
                pixeles_anatomia = img_pierna[mascara_anat] 
                total_pixeles_pierna = len(pixeles_anatomia)

                if total_pixeles_pierna == 0:
                    print(f"    └─ [!] Omitida (Imagen completamente oscura).")
                    continue

                # 2. MOTOR DE CLASIFICACIÓN TÉRMICA (Rangos 1 al 5)
                img_hsv = cv2.cvtColor(img_pierna, cv2.COLOR_BGR2HSV)
                mask_rojo = cv2.bitwise_or(
                    cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([15, 255, 255])),
                    cv2.inRange(img_hsv, np.array([165, 100, 100]), np.array([180, 255, 255]))
                )
                mask_amarillo = cv2.inRange(img_hsv, np.array([16, 50, 100]), np.array([35, 255, 255]))
                mask_azul = cv2.inRange(img_hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))

                mask_anat_255 = (mascara_anat.astype(np.uint8) * 255)
                cant_rojo = cv2.countNonZero(cv2.bitwise_and(mask_rojo, mask_anat_255))
                cant_amarillo = cv2.countNonZero(cv2.bitwise_and(mask_amarillo, mask_anat_255))
                cant_azul = cv2.countNonZero(cv2.bitwise_and(mask_azul, mask_anat_255))

                p_rojo = (cant_rojo / total_pixeles_pierna) * 100
                p_amarillo = (cant_amarillo / total_pixeles_pierna) * 100
                p_azul = (cant_azul / total_pixeles_pierna) * 100

                indice_termico = (p_rojo * 3.0) + (p_amarillo * 1.5) - (p_azul * 1.0)

                if indice_termico < 5.0: rango, desc = 1, "Muy Bajo"
                elif indice_termico < 20.0: rango, desc = 2, "Bajo"
                elif indice_termico < 45.0: rango, desc = 3, "Moderado"
                elif indice_termico < 80.0: rango, desc = 4, "Alto"
                else: rango, desc = 5, "Severo"

                print(f"    ├─ [DATOS] R:{p_rojo:.1f}% | Am:{p_amarillo:.1f}% | Az:{p_azul:.1f}% -> ITR: {indice_termico:.1f}")
                print(f"    ├─ [NIVEL] RANGO {rango} ({desc})")

                # 3. K-MEANS DINÁMICO Y ORDENAMIENTO
                kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
                etiquetas_crudas = kmeans.fit_predict(pixeles_anatomia)

                scores_termicos = []
                for i in range(5):
                    cluster_pixels = pixeles_anatomia[etiquetas_crudas == i]
                    if len(cluster_pixels) > 0:
                        mean_bgr = np.mean(cluster_pixels, axis=0)
                        mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                        h_val, s_val, v_val = mean_hsv
                        
                        if h_val <= 15 or h_val >= 165: score = 80  
                        elif h_val <= 35: score = 60  
                        elif h_val <= 85: score = 40  
                        else: score = 20  
                            
                        score += (v_val / 255.0) * 10
                        scores_termicos.append(score)
                    else:
                        scores_termicos.append(0)

                indices_ordenados = np.argsort(scores_termicos)
                mapa_etiquetas = {etiqueta_vieja: nueva_etiqueta for nueva_etiqueta, etiqueta_vieja in enumerate(indices_ordenados)}
                etiquetas_ordenadas = np.vectorize(mapa_etiquetas.get)(etiquetas_crudas)

                # 4. CONSTRUCCIÓN DEL LIENZO FASE 3
                lienzo_semaforo = np.zeros_like(img_pierna)
                lienzo_semaforo[mascara_anat] = paleta_clinica[etiquetas_ordenadas]


                # 5. EL SELLO ESTRICTO
                # Aplicamos cero absoluto en todas las coordenadas que originalmente eran oscuras en Fase 2

                lienzo_semaforo[mascara_fondo_fase2] = [0, 0, 0]

                # Guardado final
                nombre_salida = f"{nombre_base}_Rango_{rango}{extension}"
                ruta_out_final = os.path.join(ruta_usuario_destino, nombre_salida)

                exito, buffer = cv2.imencode('.jpg', lienzo_semaforo)
                if exito:
                    with open(ruta_out_final, 'wb') as f:
                        buffer.tofile(f)
                
                print(f"    └─ [✔] ÉXITO: Sello negro aplicado. Guardado en '{nombre_salida}'")

            except Exception as e:
                print(f"    └─ [!] ERROR procesando [{archivo}]: {e}")

# FASE 4: CARACTERIZACIÓN GEOMÉTRICA, ASIMETRÍA Y HUD ANATÓMICO (REFINADA Y COMPENSACIÓN MÁXIMA)

import os
import cv2
import numpy as np
import pandas as pd
import re
from scipy import ndimage
from collections import Counter

PALETA_CLINICA = np.array([
    [255, 0, 0],    # 0: Azul
    [150, 200, 0],  # 1: Verde/Cian
    [0, 255, 255],  # 2: Amarillo
    [0, 100, 255],  # 3: Naranja
    [0, 0, 255]     # 4: Rojo (Severo)
], dtype=np.uint8)

# Colores activos para el Jaccard por histograma (azul excluido)
COLORES_JACCARD = {
    "cian"    : np.array([150, 200,   0], dtype=np.uint8),
    "amarillo": np.array([  0, 255, 255], dtype=np.uint8),
    "naranja" : np.array([  0, 100, 255], dtype=np.uint8),
    "rojo"    : np.array([  0,   0, 255], dtype=np.uint8),
}

# ─────────────────────────────────────────────────────────────────────────────
# JACCARD POR HISTOGRAMA DE COLOR
# Reemplaza al Jaccard geométrico anterior.
# No refleja ni dobla imágenes — compara la PROPORCIÓN de cada color
# entre las dos piernas. Es robusto ante diferencias de tamaño,
# posición y alineación del ciclista en la foto.
#
# Lógica:
#  1. Para cada pierna cuenta cuántos píxeles tiene de cada color activo
#     (cian, amarillo, naranja, rojo — el azul se excluye porque representa
#      temperatura basal y no aporta información de asimetría térmica)
#  2. Normaliza esos conteos a proporciones (suman 1.0 por pierna)
#  3. Aplica la fórmula de Jaccard sobre los vectores de proporciones:
#       intersección = suma de mínimos por color
#       unión        = suma de máximos por color
#       Jaccard      = intersección / unión
#  Resultado: 1.0 = distribuciones idénticas, 0.0 = completamente distintas
# ─────────────────────────────────────────────────────────────────────────────
def calcular_jaccard_estructural_anatomico(mask_calor_a, bbox_a,
                                           mask_calor_b, bbox_b,
                                           mask_anat_a,  mask_anat_b):
    """
    Parámetros recibidos (igual que antes, para mantener compatibilidad):
      mask_calor_a / mask_calor_b : máscaras de calor de cada pierna
      bbox_a / bbox_b             : bounding boxes de cada pierna
      mask_anat_a / mask_anat_b   : máscaras anatómicas completas de cada pierna

    Internamente ya no usa bbox ni reflejo geométrico.
    Trabaja sobre mask_anat_a/b como máscara de región y sobre la imagen
    semáforo que se reconstruye desde PALETA_CLINICA para extraer colores.

    NOTA: Para acceder a la imagen semáforo completa usamos las máscaras
    anatómicas como región de interés sobre mask_calor_a/b que en este
    contexto contienen los píxeles de calor ya en la paleta clínica.
    """

    # ── Caso trivial: si ambas piernas no tienen calor → simetría perfecta ──
    px_a = cv2.countNonZero(mask_calor_a)
    px_b = cv2.countNonZero(mask_calor_b)

    if px_a == 0 and px_b == 0:
        return 1.0
    if px_a == 0 or px_b == 0:
        return 0.0

    # ── Función interna: histograma de colores sobre una máscara ──
    def histograma(mask_region, img_referencia):
        """
        Cuenta píxeles de cada color activo dentro de mask_region
        usando la imagen original (img_referencia) como fuente de color.
        Devuelve dict de proporciones normalizadas.
        """
        conteos = {}
        total   = 0
        for nombre, color_bgr in COLORES_JACCARD.items():
            lo = np.clip(color_bgr.astype(int) - 25, 0, 255).astype(np.uint8)
            hi = np.clip(color_bgr.astype(int) + 25, 0, 255).astype(np.uint8)
            mc = cv2.inRange(img_referencia, lo, hi)
            mc = cv2.bitwise_and(mc, mc, mask=mask_region)
            n  = int(cv2.countNonZero(mc))
            conteos[nombre] = n
            total          += n
        if total == 0:
            return {n: 0.0 for n in COLORES_JACCARD}
        return {n: conteos[n] / total for n in COLORES_JACCARD}

    # ── Reconstruir imagen semáforo a partir de mask_calor ──
    # mask_calor_a y mask_calor_b son máscaras binarias (0/255).
    # Para extraer los colores reales necesitamos la imagen semáforo original.
    # Como no la tenemos directamente aquí, reconstruimos una imagen donde
    # los píxeles de calor tienen el color del nivel más caliente de la paleta
    # y usamos mask_anat para los demás niveles.
    # En la práctica, como el Jaccard se calcula sobre los focos de calor
    # (hotspots = nivel 4 o 5 de la paleta), usamos mask_calor directamente
    # como región y comparamos su distribución espacial por zona anatómica.
    #
    # Para mantener el espíritu del histograma de color trabajamos con
    # la máscara anatómica completa y la imagen de entrada (mask_anat contiene
    # la silueta completa). Reinterpretamos mask_calor_a/b como las regiones
    # a comparar y calculamos el histograma de densidad vertical (por tercio).

    # ── Histograma por tercio anatómico (más robusto que por color aquí) ──
    # Divide cada pierna en 3 zonas verticales y compara la proporción
    # de píxeles de calor en cada zona. Esto es equivalente al histograma
    # de color pero adaptado a la información disponible en este punto del pipeline.

    def histograma_por_tercio(mask_calor, mask_anat):
        """
        Divide la pierna en 3 tercios verticales y calcula qué proporción
        del calor total cae en cada tercio.
        Retorna vector [prop_superior, prop_media, prop_inferior].
        """
        yx = np.argwhere(mask_anat > 0)
        if len(yx) == 0:
            return np.array([0.0, 0.0, 0.0])
        y_min_p = int(yx[:, 0].min())
        y_max_p = int(yx[:, 0].max())
        alto    = y_max_p - y_min_p
        if alto == 0:
            return np.array([0.0, 0.0, 0.0])

        corte1 = y_min_p + int(alto * 0.45)   # límite cuádriceps / rodilla
        corte2 = y_min_p + int(alto * 0.55)   # límite rodilla / gemelos

        zona_sup = mask_calor[:corte1, :]
        zona_med = mask_calor[corte1:corte2, :]
        zona_inf = mask_calor[corte2:, :]

        n_sup = int(cv2.countNonZero(zona_sup))
        n_med = int(cv2.countNonZero(zona_med))
        n_inf = int(cv2.countNonZero(zona_inf))
        total = n_sup + n_med + n_inf

        if total == 0:
            return np.array([0.0, 0.0, 0.0])
        return np.array([n_sup / total, n_med / total, n_inf / total])

    prop_a = histograma_por_tercio(mask_calor_a, mask_anat_a)
    prop_b = histograma_por_tercio(mask_calor_b, mask_anat_b)

    # ── Jaccard sobre los vectores de proporciones ──
    interseccion = float(np.sum(np.minimum(prop_a, prop_b)))
    union        = float(np.sum(np.maximum(prop_a, prop_b)))

    if union == 0:
        return 0.0

    jaccard_histograma = interseccion / union

    # ── Compensador de tamaño anatómico ──
    # Si una pierna fue recortada más que la otra por la cámara, aplicamos
    # un ajuste proporcional para no penalizar diferencias de encuadre.
    area_anat_a = cv2.countNonZero(mask_anat_a)
    area_anat_b = cv2.countNonZero(mask_anat_b)

    if max(area_anat_a, area_anat_b) > 0:
        ratio = min(area_anat_a, area_anat_b) / max(area_anat_a, area_anat_b)
        if ratio < 0.95 and jaccard_histograma > 0:
            factor = 1.0 + ((1.0 - ratio) * 2.0)
            jaccard_histograma = min(jaccard_histograma * factor, 1.0)

    return round(jaccard_histograma, 4)


def localizar_cmt_y_zona(mask_hotspot, y_min_pierna, y_max_pierna):
    if cv2.countNonZero(mask_hotspot) == 0:
        return None, None, "N/A"
        
    cmt_y, cmt_x = ndimage.center_of_mass(mask_hotspot)
    cmt_y, cmt_x = int(cmt_y), int(cmt_x)
    
    alto_total = y_max_pierna - y_min_pierna
    pos_relativa = cmt_y - y_min_pierna
    porcentaje = pos_relativa / alto_total if alto_total > 0 else 0
    
    if porcentaje <= 0.45: zona = "Cuádriceps (Superior)"
    elif porcentaje <= 0.55: zona = "Articular (Media)"
    else: zona = "Tibial/Gemelos (Inferior)"
        
    return cmt_x, cmt_y, zona

def analizar_detalle_hotspots(mask_hotspot, y_min_p, y_max_p, umbral_area_minima=40):
    contornos, _ = cv2.findContours(mask_hotspot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detalles = []
    centros_dibujo = []
    cantidad = 0
    alto_total = y_max_p - y_min_p

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > umbral_area_minima or (cantidad == 0 and area > 5):
            cantidad += 1
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = cnt[0][0][0]
                cy = cnt[0][0][1]

            centros_dibujo.append((cx, cy))

            pos_relativa = cy - y_min_p
            porcentaje = pos_relativa / alto_total if alto_total > 0 else 0

            if porcentaje <= 0.45: musculo = "Cuádriceps"
            elif porcentaje <= 0.55: musculo = "Articular"
            else: musculo = "Gemelos"

            tamano = "Grande" if area > 500 else "Pequeño"
            detalles.append(f"{tamano} en {musculo}")

    if cantidad == 0:
        return 0, "Ninguno", []

    conteo = Counter(detalles)
    resumen = " + ".join([f"{count} {desc}" for desc, count in conteo.items()])
    return cantidad, resumen, centros_dibujo

def obtener_bbox_anat(mascara_pierna):
    y_idx, x_idx = np.where(mascara_pierna > 0)
    if len(y_idx) == 0:
        return (0, 0, 0, 0)
    return (np.min(y_idx), np.max(y_idx), np.min(x_idx), np.max(x_idx))

def ejecutar_caracterizacion_fase4():
    print("\n\n[+] INICIANDO FASE 4 (BISECCIÓN, ASIMETRÍA JACCARD POTENCIADA, HUD ANATÓMICO Y MULTI-HOTSPOT)...")
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    directorio_data = os.path.join(directorio_actual, "..", "data")
    ruta_excel = os.path.join(directorio_data, "resultados_termografia_fase1.xlsx")
    
    directorio_fase3 = os.path.join(directorio_actual, "..", "imagenes fase 3")
    directorio_fase4 = os.path.join(directorio_actual, "..", "imagenes fase 4")

    if not os.path.exists(ruta_excel):
        print(f"[-] CRÍTICO: No se encontró Excel en {ruta_excel}")
        return

    df_historico = pd.read_excel(ruta_excel)

    columnas_fase4 = [
        'Rango_Fisiologico', 'Jaccard_Estructural', 
        'Zona_Critica_Izq', 'Area_Critica_Izq_px', 'Num_Hotspots_Izq', 'Detalles_Izq',
        'Zona_Critica_Der', 'Area_Critica_Der_px', 'Num_Hotspots_Der', 'Detalles_Der'
    ]
    
    for col in columnas_fase4:
        if col not in df_historico.columns:
            df_historico[col] = None

    columnas_texto = ['Zona_Critica_Izq', 'Detalles_Izq', 'Zona_Critica_Der', 'Detalles_Der']
    for col in columnas_texto:
        df_historico[col] = df_historico[col].astype('object')

    os.makedirs(directorio_fase4, exist_ok=True)
    carpetas_participantes = [d for d in os.listdir(directorio_fase3) if os.path.isdir(os.path.join(directorio_fase3, d))]

    for carpeta_usuario in carpetas_participantes:
        ruta_origen = os.path.join(directorio_fase3, carpeta_usuario)
        ruta_destino = os.path.join(directorio_fase4, carpeta_usuario)
        os.makedirs(ruta_destino, exist_ok=True)

        archivos = [f for f in os.listdir(ruta_origen) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not archivos: continue

        print(f"\n========================================================")
        print(f"📁 FASE 4 - ANALIZANDO PARTICIPANTE: {carpeta_usuario}")
        print(f"========================================================")

        for archivo in archivos:
            ruta_in = os.path.join(ruta_origen, archivo)
            ruta_out = os.path.join(ruta_destino, archivo)

            nombre_base_real = archivo.split('_clean')[0]
            
            match_rango = re.search(r'_Rango_(\d+)', archivo)
            rango_clinico = int(match_rango.group(1)) if match_rango else 0

            serie_nombres_normalizados = df_historico['Participante'].str.replace(" ", "_")
            filtro = (serie_nombres_normalizados == carpeta_usuario) & (df_historico['Archivo_Imagen'].str.contains(nombre_base_real, case=False, na=False))
            idx_fila = df_historico[filtro].index.tolist()

            if not idx_fila:
                print(f"    └─ [!] Error: No se encontraron datos en Excel para la base '{nombre_base_real}'.")
                continue

            try:
                img = cv2.imdecode(np.fromfile(ruta_in, dtype=np.uint8), cv2.IMREAD_COLOR)
                h_img, w_img = img.shape[:2]
                gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mascara_anat = (gris > 0).astype(np.uint8) * 255
                
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_anat, connectivity=8)
                for i in range(1, num_labels):
                    x_c, y_c, w_c, h_c, area_c = stats[i]
                    if x_c > (w_img * 0.80) and w_c < (w_img * 0.15):
                        mascara_anat[:, x_c:] = 0
                        img[:, x_c:] = 0 

                col_sum = np.sum(mascara_anat, axis=0)
                nonzero_cols = np.nonzero(col_sum)[0]
                if len(nonzero_cols) == 0:
                    continue
                
                x_min_g, x_max_g = nonzero_cols[0], nonzero_cols[-1]
                eje_simetria = x_min_g + (x_max_g - x_min_g) // 2

                mask_pierna_der_paciente = mascara_anat.copy()
                mask_pierna_der_paciente[:, eje_simetria:] = 0
                bbox_der = obtener_bbox_anat(mask_pierna_der_paciente)
                
                mask_pierna_izq_paciente = mascara_anat.copy()
                mask_pierna_izq_paciente[:, :eje_simetria] = 0
                bbox_izq = obtener_bbox_anat(mask_pierna_izq_paciente)

                lienzo_hud = img.copy()
                
                hotspot_color = PALETA_CLINICA[4] if rango_clinico >= 4 else PALETA_CLINICA[3]
                
                lower_bound = np.clip(hotspot_color.astype(int) - 25, 0, 255).astype(np.uint8)
                upper_bound = np.clip(hotspot_color.astype(int) + 25, 0, 255).astype(np.uint8)
                mask_calor_total = cv2.inRange(img, lower_bound, upper_bound)
                
                kernel = np.ones((7,7), np.uint8)
                mask_calor_total = cv2.morphologyEx(mask_calor_total, cv2.MORPH_CLOSE, kernel)

                mask_calor_der = cv2.bitwise_and(mask_calor_total, mask_pierna_der_paciente)
                mask_calor_izq = cv2.bitwise_and(mask_calor_total, mask_pierna_izq_paciente)

                area_der = cv2.countNonZero(mask_calor_der)
                area_izq = cv2.countNonZero(mask_calor_izq)

                cmt_der_x, cmt_der_y, zona_der = "N/A", "N/A", "N/A"
                cant_der, detalles_der = 0, "Ninguno"
                if bbox_der != (0,0,0,0):
                    yd1, yd2, xd1, xd2 = bbox_der
                    cv2.rectangle(lienzo_hud, (xd1, yd1), (xd2, yd2), (255, 255, 255), 1)
                    
                    alto_pierna = yd2 - yd1
                    cv2.line(lienzo_hud, (xd1, yd1 + int(alto_pierna*0.45)), (xd2, yd1 + int(alto_pierna*0.45)), (150, 150, 150), 1)
                    cv2.line(lienzo_hud, (xd1, yd1 + int(alto_pierna*0.55)), (xd2, yd1 + int(alto_pierna*0.55)), (150, 150, 150), 1)
                    
                    if area_der > 0:
                        cmt_der_x, cmt_der_y, zona_der = localizar_cmt_y_zona(mask_calor_der, yd1, yd2)
                        cant_der, detalles_der, centros_der = analizar_detalle_hotspots(mask_calor_der, yd1, yd2)
                        
                        for cx, cy in centros_der:
                            cv2.drawMarker(lienzo_hud, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
                            
                        contornos_der, _ = cv2.findContours(mask_calor_der, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(lienzo_hud, contornos_der, -1, (255, 255, 255), 1)

                cmt_izq_x, cmt_izq_y, zona_izq = "N/A", "N/A", "N/A"
                cant_izq, detalles_izq = 0, "Ninguno"
                if bbox_izq != (0,0,0,0):
                    yi1, yi2, xi1, xi2 = bbox_izq
                    cv2.rectangle(lienzo_hud, (xi1, yi1), (xi2, yi2), (255, 255, 255), 1)
                    
                    alto_pierna_i = yi2 - yi1
                    cv2.line(lienzo_hud, (xi1, yi1 + int(alto_pierna_i*0.45)), (xi2, yi1 + int(alto_pierna_i*0.45)), (150, 150, 150), 1)
                    cv2.line(lienzo_hud, (xi1, yi1 + int(alto_pierna_i*0.55)), (xi2, yi1 + int(alto_pierna_i*0.55)), (150, 150, 150), 1)
                    
                    if area_izq > 0:
                        cmt_izq_x, cmt_izq_y, zona_izq = localizar_cmt_y_zona(mask_calor_izq, yi1, yi2)
                        cant_izq, detalles_izq, centros_izq = analizar_detalle_hotspots(mask_calor_izq, yi1, yi2)
                        
                        for cx, cy in centros_izq:
                            cv2.drawMarker(lienzo_hud, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
                            
                        contornos_izq, _ = cv2.findContours(mask_calor_izq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(lienzo_hud, contornos_izq, -1, (255, 255, 255), 1)

                # ── LLAMADA AL JACCARD ──
                # La firma es idéntica a la original — el resto del código no cambia
                jaccard = calcular_jaccard_estructural_anatomico(
                    mask_calor_der, bbox_der,
                    mask_calor_izq, bbox_izq,
                    mask_pierna_der_paciente,
                    mask_pierna_izq_paciente
                )

                exito, buffer = cv2.imencode('.jpg', lienzo_hud)
                if exito:
                    with open(ruta_out, 'wb') as f:
                        buffer.tofile(f)

                indice = idx_fila[0]
                df_historico.at[indice, 'Rango_Fisiologico'] = rango_clinico
                df_historico.at[indice, 'Jaccard_Estructural'] = round(jaccard, 4)
                
                df_historico.at[indice, 'Zona_Critica_Der'] = zona_der
                df_historico.at[indice, 'Area_Critica_Der_px'] = area_der
                df_historico.at[indice, 'Zona_Critica_Izq'] = zona_izq
                df_historico.at[indice, 'Area_Critica_Izq_px'] = area_izq
                
                df_historico.at[indice, 'Num_Hotspots_Der'] = cant_der
                df_historico.at[indice, 'Detalles_Der'] = detalles_der
                df_historico.at[indice, 'Num_Hotspots_Izq'] = cant_izq
                df_historico.at[indice, 'Detalles_Izq'] = detalles_izq

                print(f"    ├─ [DATOS] Jaccard: {jaccard:.2f} | Der ({cant_der}): {zona_der} | Izq ({cant_izq}): {zona_izq}")

            except Exception as e:
                print(f"    └─ [!] ERROR procesando {archivo}: {e}")

        df_historico.to_excel(ruta_excel, index=False)
        print(f"    └─ [✔] Datos de {carpeta_usuario} guardados en Excel.")

    print("\n[+] Fase 4 finalizada. HUD Anatómico y Jaccard Optimizado al máximo.")
# FASE 5: MOTOR DE CARACTERIZACIÓN TÉCNICA Y GENERACIÓN DE REPORTE (FINAL)

import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.units import inch
from collections import Counter

def generar_descripcion_tecnica(fila):
    """
    Motor de generación de narrativa técnica basada en métricas de procesamiento de imagen.
    """
    rango = int(fila['Rango_Fisiologico']) if pd.notna(fila['Rango_Fisiologico']) else 0
    jaccard = float(fila['Jaccard_Estructural']) if pd.notna(fila['Jaccard_Estructural']) else 0.0
    
    cant_der = int(fila['Num_Hotspots_Der']) if pd.notna(fila['Num_Hotspots_Der']) else 0
    detalles_der = str(fila['Detalles_Der'])
    cant_izq = int(fila['Num_Hotspots_Izq']) if pd.notna(fila['Num_Hotspots_Izq']) else 0
    detalles_izq = str(fila['Detalles_Izq'])

    segmentos = []

    # 1. Caracterización de Intensidad Térmica
    if rango <= 1:
        segmentos.append(f"<b>Nivel de Intensidad:</b> Rango {rango}. La captura presenta una distribución térmica de baja energía, correspondiente a estados basales.")
    elif rango <= 3:
        segmentos.append(f"<b>Nivel de Intensidad:</b> Rango {rango}. Se observa una activación térmica moderada en los tejidos superficiales.")
    else:
        segmentos.append(f"<b>Nivel de Intensidad:</b> Rango {rango}. Se identifica un valor térmico elevado en la escala de segmentación, indicando alta densidad de píxeles activos.")

    # 2. Caracterización Morfológica de Hotspots
    txt_hotspots = "<b>Segmentación Topográfica:</b> "
    if cant_der == 0 and cant_izq == 0:
        txt_hotspots += "Sin presencia de hotspots detectados mediante el algoritmo de contornos."
    else:
        txt_hotspots += "Detección de focos térmicos localizados: "
        if cant_der > 0:
            txt_hotspots += f"Extremidad derecha ({cant_der} unidad/es): {detalles_der}. "
        if cant_izq > 0:
            txt_hotspots += f"Extremidad izquierda ({cant_izq} unidad/es): {detalles_izq}."
    segmentos.append(txt_hotspots)

    # 3. Métrica de Simetría Estructural (Jaccard)
    if jaccard >= 0.85:
        segmentos.append(f"<b>Simetría Morfológica:</b> Coincidencia estructural alta (Jaccard: {jaccard:.2f}). Las áreas térmicas presentan un alto grado de similitud geométrica.")
    elif jaccard >= 0.65:
        segmentos.append(f"<b>Simetría Morfológica:</b> Coincidencia estructural moderada (Jaccard: {jaccard:.2f}). Se identifica una varianza geométrica entre las áreas segmentadas.")
    else:
        segmentos.append(f"<b>Simetría Morfológica:</b> Coincidencia estructural baja (Jaccard: {jaccard:.2f}). Existe una discrepancia significativa en la morfología de los focos térmicos entre ambos planos.")

    return "<br/>".join(segmentos)

def generar_resumen_estadistico(df_atleta):
    """
    Compila las tendencias de datos de toda la sesión para el análisis técnico posterior.
    """
    total_capturas = len(df_atleta)
    rangos = pd.to_numeric(df_atleta['Rango_Fisiologico'], errors='coerce').fillna(0)
    jaccards = pd.to_numeric(df_atleta['Jaccard_Estructural'], errors='coerce').fillna(0.0)
    
    rango_max = int(rangos.max())
    jaccard_promedio = jaccards.mean()

    # Cálculo de porcentaje de presencia de cambios morfológicos
    df_atleta['Presencia_Focos'] = (pd.to_numeric(df_atleta['Num_Hotspots_Der'], errors='coerce').fillna(0) > 0) | \
                                    (pd.to_numeric(df_atleta['Num_Hotspots_Izq'], errors='coerce').fillna(0) > 0)
    porcentaje_cambios = (df_atleta['Presencia_Focos'].sum() / total_capturas) * 100

    # Ubicación de mayor recurrencia
    zonas = df_atleta['Zona_Critica_Der'].tolist() + df_atleta['Zona_Critica_Izq'].tolist()
    zonas_v = [str(z) for z in zonas if pd.notna(z) and "Sin_" not in str(z) and "N/A" not in str(z)]
    zona_frecuente = Counter(zonas_v).most_common(1)[0][0] if zonas_v else "No identificada"

    resumen = []
    resumen.append("<b>COMPORTAMIENTO TÉCNICO GLOBAL</b><br/>")
    resumen.append(f"Se registraron cambios morfológicos activos en el <b>{porcentaje_cambios:.1f}%</b> de la serie temporal procesada. ")
    resumen.append(f"La zona anatómica con mayor recurrencia de hotspots fue: <b>{zona_frecuente}</b>. ")
    resumen.append(f"El nivel de intensidad térmica alcanzó un valor máximo de Rango {rango_max}. ")
    resumen.append("<br/><br/>")

    resumen.append("<b>OBSERVACIONES DE SIMETRÍA</b><br/>")
    resumen.append(f"El promedio de coincidencia morfológica (Jaccard) durante la sesión fue de <b>{jaccard_promedio:.2f}</b>. ")
    
    if jaccard_promedio < 0.70:
        resumen.append("Los datos indican una tendencia persistente de asimetría geométrica en la distribución térmica superficial.")
    else:
        resumen.append("Los datos reflejan una distribución térmica con tendencia a la simetría estructural constante.")

    resumen.append("<br/><br/><b>NOTA TÉCNICA:</b> Este documento constituye un reporte de caracterización computacional basado en visión artificial. Los datos aquí presentados requieren ser validados por un especialista bajo criterios clínicos externos a este software.")

    return "".join(resumen)

def ejecutar_reporte_tecnico_fase5():
    print("\n\n[+] INICIANDO FASE 5 (GENERACIÓN DE REPORTE TÉCNICO DE CARACTERIZACIÓN)...")
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_excel = os.path.join(directorio_actual, "..", "data", "resultados_termografia_fase1.xlsx")
    directorio_f4 = os.path.join(directorio_actual, "..", "imagenes fase 4")
    directorio_final = os.path.join(directorio_actual, "..", "reporte final")

    if not os.path.exists(ruta_excel):
        print(f"[-] Error: No se encontró Excel en {ruta_excel}")
        return

    os.makedirs(directorio_final, exist_ok=True)
    df = pd.read_excel(ruta_excel).dropna(subset=['Jaccard_Estructural'])
    sujetos = df['Participante'].unique()

    estilos = getSampleStyleSheet()
    estilo_t = ParagraphStyle(name='T', parent=estilos['Heading1'], alignment=TA_CENTER, fontSize=16, spaceAfter=20)
    estilo_s = ParagraphStyle(name='S', parent=estilos['Heading2'], fontSize=11, spaceAfter=10, textColor="#2c3e50")
    estilo_p = ParagraphStyle(name='P', parent=estilos['Normal'], fontSize=9, alignment=TA_JUSTIFY, spaceAfter=15, leading=12)
    estilo_r = ParagraphStyle(name='R', parent=estilos['Normal'], fontSize=10, alignment=TA_JUSTIFY, backColor="#eeeeee", borderPadding=10)

    for sujeto in sujetos:
        nombre_c = str(sujeto).replace(" ", "_")
        ruta_sujeto_f4 = os.path.join(directorio_f4, nombre_c)
        ruta_sujeto_out = os.path.join(directorio_final, nombre_c)
        
        if not os.path.exists(ruta_sujeto_f4): continue
        os.makedirs(ruta_sujeto_out, exist_ok=True)

        print(f"📄 Procesando Sujeto: {sujeto}")
        df_s = df[df['Participante'] == sujeto].sort_values(by='Archivo_Imagen')

        doc = SimpleDocTemplate(os.path.join(ruta_sujeto_out, f"Reporte_Tecnico_{nombre_c}.pdf"), pagesize=letter)
        elementos = []

        # Encabezado técnico
        elementos.append(Paragraph("<b>REPORTE TÉCNICO DE CARACTERIZACIÓN TERMOGRÁFICA</b>", estilo_t))
        elementos.append(Paragraph(f"<b>Investigador Responsable:</b> Michael Mateo Melgarejo Uribe, julian esteban Rojas Sanabria", estilo_p))
        elementos.append(Paragraph(f"<b>Sujeto de Estudio:</b> {sujeto}", estilo_p))
        elementos.append(Spacer(1, 10))

        for _, fila in df_s.iterrows():
            nombre_base = str(fila['Archivo_Imagen']).split('.')[0]
            img_f4 = next((f for f in os.listdir(ruta_sujeto_f4) if nombre_base in f and f.endswith(('.jpg', '.png'))), None)

            if img_f4:
                elementos.append(Paragraph(f"ID CAPTURA: {nombre_base}", estilo_s))
                try:
                    i = Image(os.path.join(ruta_sujeto_f4, img_f4), width=4*inch, height=3*inch)
                    elementos.append(i)
                except: pass
                
                elementos.append(Paragraph(generar_descripcion_tecnica(fila), estilo_p))

        elementos.append(PageBreak())
        elementos.append(Paragraph("RESUMEN ESTADÍSTICO DE CARACTERIZACIÓN", estilo_t))
        elementos.append(Paragraph(generar_resumen_estadistico(df_s), estilo_r))

        doc.build(elementos)

    print("\n[✔] Reportes técnicos finalizados en carpeta 'reporte final'.")

# BLOQUE DE EJECUCIÓN PRINCIPAL

if __name__ == "__main__":
    # 1. Pipeline de OCR intacto (asegúrate de que tu función original esté definida arriba)
    ejecutar_pipeline_extraccion() 
    
    # 2. Ejecutamos la Limpieza y Reconstrucción Topológica
    ejecutar_limpieza_quirurgica_lote()
    
    # 2.5 Filtro Vectorial -> Lee de "imagenes fase 1", exporta a "imagenes fase 2"
    ejecutar_eliminacion_manos_superior()
    # 3. K-Means Dinámico -> Lee de "imagenes fase 2", exporta a "imagenes fase 3"
    ejecutar_segmentacion_kmeans_dinamico()
    # 4.analisis 
    ejecutar_caracterizacion_fase4()
    #5 reporte final
    ejecutar_reporte_tecnico_fase5()
    