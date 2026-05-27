# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE SENSIBILIDAD v10
# Propósito: justificar la elección de agresividad=25 (fondo) y 65 (pantaloneta)
#
# Lógica:
#  - Fondo:      rango 5-70 de 5 en 5  → 14 valores
#  - Pantaloneta: rango 30-100 de 5 en 5 → 15 valores
#  - Por cada imagen se prueban todas las combinaciones
#  - Jaccard por histograma de color (sin azul, sin doblar imágenes)
#  - Dos gráficas al final:
#      Gráfica 1: Agresividad de fondo (X) vs Jaccard promedio (Y)
#      Gráfica 2: Agresividad de pantaloneta (X) vs Jaccard promedio (Y)
#  - Líneas rojas marcando los valores usados en el sistema (25 y 65)
# ─────────────────────────────────────────────────────────────────────────────

import sys, subprocess, os

def pip(pkg, mod=None):
    try: __import__(mod or pkg)
    except ImportError:
        print(f"  Instalando {pkg}...")
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"--quiet"])

pip("opencv-python","cv2"); pip("numpy"); pip("pandas")
pip("scikit-learn","sklearn"); pip("openpyxl"); pip("scipy"); pip("matplotlib")

import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import uniform_filter1d
import tkinter as tk
from tkinter import filedialog, messagebox

# ─────────────────────────────────────────────────────────────────────────────
# RANGOS DE ANÁLISIS
# ─────────────────────────────────────────────────────────────────────────────
RANGO_FONDO       = list(range(5,  71, 5))   # 5,10,15,...,70  → 14 valores
RANGO_PANTALONETA = list(range(30, 101, 5))  # 30,35,40,...,100 → 15 valores
K_FIJO            = 5                         # K-Means fijo en 5 (igual al sistema)

# Valores usados en el sistema principal (se marcan en las gráficas)
VALOR_FONDO_USADO       = 25
VALOR_PANTALONETA_USADO = 65

PALETA_BGR = {
    "azul"    : np.array([255,   0,   0], dtype=np.uint8),
    "cian"    : np.array([150, 200,   0], dtype=np.uint8),
    "amarillo": np.array([  0, 255, 255], dtype=np.uint8),
    "naranja" : np.array([  0, 100, 255], dtype=np.uint8),
    "rojo"    : np.array([  0,   0, 255], dtype=np.uint8),
}
COLORES_ACTIVOS = ["cian", "amarillo", "naranja", "rojo"]

COLOR_BARRA = {
    "cian"    : (200, 200,   0),
    "amarillo": (  0, 220, 220),
    "naranja" : (  0, 130, 255),
    "rojo"    : (  0,   0, 220),
}

# ─────────────────────────────────────────────────────────────────────────────
# ELIMINAR BARRA LATERAL DERECHA
# ─────────────────────────────────────────────────────────────────────────────
def eliminar_barra(img_bgr):
    h, w    = img_bgr.shape[:2]
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    col_var = np.var(hsv[:,:,0].astype(float), axis=0)
    zona    = int(w * 0.75)
    umbral  = np.percentile(col_var, 85)
    hits    = np.where(col_var[zona:] > umbral)[0]
    corte   = zona + int(hits[0]) if len(hits) > 0 else int(w * 0.88)
    out     = img_bgr.copy()
    out[:, corte:] = 0
    return out, corte

# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTACIÓN EN DOS PASOS (igual que el sistema principal)
# Paso 1: umbral de fondo → máscara de todo lo que brilla más que el entorno
# Paso 2: umbral de pantaloneta → refina la mitad superior (quita la ropa)
# No usa CNN — usa umbral directo que es equivalente a lo que la CNN aprende
# con esa agresividad, pero sin el tiempo de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
def segmentar_dos_pasos(img_bgr, agr_fondo, agr_pantaloneta):
    h, w   = img_bgr.shape[:2]
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)

    # Paso 1 — separar fondo ambiental (toda la imagen)
    _, mask1 = cv2.threshold(gray, agr_fondo, 255, cv2.THRESH_BINARY)
    mask1    = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    mask1    = cv2.morphologyEx(mask1, cv2.MORPH_OPEN,  kernel)

    # Paso 2 — refinar mitad superior (eliminar pantaloneta)
    mask2            = mask1.copy()
    gray_sup         = gray[0:h//2, :]
    _, mask_sup      = cv2.threshold(gray_sup, agr_pantaloneta, 255, cv2.THRESH_BINARY)
    mask_sup         = cv2.morphologyEx(mask_sup, cv2.MORPH_CLOSE, kernel)
    mask2[0:h//2, :] = mask_sup

    # Limpieza final — eliminar componentes pequeños
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(mask2)
    for i in range(1, nl):
        if stats[i, cv2.CC_STAT_AREA] < 500:
            mask2[labels == i] = 0

    return mask2

# ─────────────────────────────────────────────────────────────────────────────
# K-MEANS EN HSV (K fijo = 5)
# ─────────────────────────────────────────────────────────────────────────────
def kmeans_semaforo(img_bgr, mask):
    mb = mask > 0
    px = img_bgr[mb]
    if len(px) < 50:
        return np.zeros_like(img_bgr)
    px_hsv = cv2.cvtColor(px.reshape(1,-1,3).astype(np.uint8),
                          cv2.COLOR_BGR2HSV).reshape(-1,3).astype(np.float32)
    km  = KMeans(n_clusters=K_FIJO, n_init=10, random_state=42)
    etq = km.fit_predict(px_hsv)
    scores = []
    for i in range(K_FIJO):
        if (etq==i).sum() > 0:
            hv = km.cluster_centers_[i][0]
            vv = km.cluster_centers_[i][2]
            s  = (80 if (hv<=15 or hv>=165) else
                  60 if hv<=35 else 40 if hv<=85 else 20)
            s += (vv/255.0)*10
        else:
            s = 0
        scores.append(s)
    idx_ord = np.argsort(scores)
    mapa    = {v: i for i, v in enumerate(idx_ord)}
    nivel   = np.minimum(np.vectorize(mapa.get)(etq), 4)
    paleta  = np.array([PALETA_BGR[n] for n in
                        ["azul","cian","amarillo","naranja","rojo"]], dtype=np.uint8)
    lienzo  = np.zeros_like(img_bgr)
    lienzo[mb] = paleta[nivel]
    return lienzo

# ─────────────────────────────────────────────────────────────────────────────
# SEPARAR PIERNAS — valle de densidad en zona central
# ─────────────────────────────────────────────────────────────────────────────
def separar_piernas(mask):
    h, w    = mask.shape
    col_sum = np.sum(mask > 0, axis=0).astype(float)
    if col_sum.sum() == 0:
        return None, None, -1, "Máscara vacía"
    ini, fin = int(w*0.30), int(w*0.70)
    zona     = col_sum[ini:fin]
    if zona.max() == 0:
        ini, fin = int(w*0.20), int(w*0.80)
        zona = col_sum[ini:fin]
    if zona.max() == 0:
        return None, None, -1, "Sin píxeles en zona central"
    zona_s = uniform_filter1d(zona, size=max(5, int(len(zona)*0.08)))
    eje    = int(np.argmin(zona_s)) + ini
    md = mask.copy(); md[:, eje:] = 0
    mi = mask.copy(); mi[:, :eje] = 0
    pd_ = int(np.sum(md > 0)); pi = int(np.sum(mi > 0))
    if pd_ == 0 or pi == 0:
        cols = np.where(col_sum > 0)[0]
        if len(cols) == 0:
            return None, None, -1, "Sin columnas activas"
        eje = int((cols[0]+cols[-1])/2)
        md = mask.copy(); md[:, eje:] = 0
        mi = mask.copy(); mi[:, :eje] = 0
        pd_ = int(np.sum(md>0)); pi = int(np.sum(mi>0))
        if pd_==0 or pi==0:
            return None, None, -1, f"Fallo en centro x={eje}"
    return md, mi, eje, f"OK der={pd_}px izq={pi}px eje={eje}"

# ─────────────────────────────────────────────────────────────────────────────
# HISTOGRAMA DE COLOR Y JACCARD POR HISTOGRAMA
# ─────────────────────────────────────────────────────────────────────────────
def histograma_color(img_sem, mask_pierna):
    conteos = {}
    total   = 0
    for nombre in COLORES_ACTIVOS:
        color = PALETA_BGR[nombre]
        lo    = np.clip(color.astype(int)-25, 0, 255).astype(np.uint8)
        hi    = np.clip(color.astype(int)+25, 0, 255).astype(np.uint8)
        mc    = cv2.inRange(img_sem, lo, hi)
        mc    = cv2.bitwise_and(mc, mc, mask=mask_pierna)
        n     = int(cv2.countNonZero(mc))
        conteos[nombre] = n
        total          += n
    if total == 0:
        return {n: 0.0 for n in COLORES_ACTIVOS}, conteos, 0
    return {n: conteos[n]/total for n in COLORES_ACTIVOS}, conteos, total

def jaccard_histograma(prop_a, prop_b):
    inter = sum(min(prop_a[n], prop_b[n]) for n in COLORES_ACTIVOS)
    union = sum(max(prop_a[n], prop_b[n]) for n in COLORES_ACTIVOS)
    return float(inter/union) if union > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN EN TIEMPO REAL
# ─────────────────────────────────────────────────────────────────────────────
def mostrar(archivo, agr_f, agr_p,
            img_orig, img_sb, mask, img_sem,
            md, mi, eje, jac_val,
            prop_der, prop_izq, cnt_der, cnt_izq,
            sep_info, barra_x):

    h, w = img_orig.shape[:2]
    TH   = 320

    def r(im):
        sc = TH / im.shape[0]
        return cv2.resize(im, (int(im.shape[1]*sc), TH))

    # Panel 1 — original
    p1 = img_orig.copy()
    cv2.line(p1, (barra_x,0),(barra_x,h),(0,255,255),2)
    cv2.putText(p1,"barra cortada",(max(0,barra_x-100),18),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,255,255),1)

    # Panel 2 — máscara con eje
    p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if eje > 0:
        cv2.line(p2,(eje,0),(eje,h),(0,255,0),2)
    cv2.putText(p2,f"f={agr_f} p={agr_p}",(5,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.44,(0,255,255),1)
    cv2.putText(p2,sep_info[:38],(5,42),
                cv2.FONT_HERSHEY_SIMPLEX,0.33,(255,255,0),1)
    # Línea divisoria entre zona fondo y zona pantaloneta
    cv2.line(p2,(0,h//2),(w,h//2),(200,100,200),1)
    cv2.putText(p2,"^ pantaloneta",(5,h//2-4),
                cv2.FONT_HERSHEY_SIMPLEX,0.30,(200,100,200),1)

    # Panel 3 — semáforo con contornos
    p3 = img_sem.copy()
    if md is not None:
        ct,_ = cv2.findContours(md,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p3,ct,-1,(255,255,255),1)
    if mi is not None:
        ct,_ = cv2.findContours(mi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p3,ct,-1,(180,180,180),1)
    if eje > 0:
        cv2.line(p3,(eje,0),(eje,h),(0,255,255),2)

    # Panel 4 — barras de histograma comparativo
    BAR_H, BAR_W = TH, 280
    p4 = np.zeros((BAR_H,BAR_W,3),dtype=np.uint8)
    p4[:] = (30,30,45)
    cv2.putText(p4,f"Jaccard = {jac_val:.4f}",(8,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1)
    cv2.putText(p4,"DER=llena  IZQ=borde oscuro",(8,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.30,(160,160,160),1)

    bar_x0   = 8
    bar_maxw = BAR_W - 16
    row_h    = 52
    y_start  = 50

    for i, nombre in enumerate(COLORES_ACTIVOS):
        y0   = y_start + i*row_h
        pd_v = float(prop_der.get(nombre,0.0))
        pi_v = float(prop_izq.get(nombre,0.0))
        cd_v = int(cnt_der.get(nombre,0))
        ci_v = int(cnt_izq.get(nombre,0))
        c    = COLOR_BARRA[nombre]
        c_dim= (c[0]//2, c[1]//2, c[2]//2)
        bw_d = int(pd_v * bar_maxw)
        bw_i = int(pi_v * bar_maxw)
        if bw_d > 0:
            cv2.rectangle(p4,(bar_x0,y0),(bar_x0+bw_d,y0+16),c,-1)
        if bw_i > 0:
            cv2.rectangle(p4,(bar_x0,y0+20),(bar_x0+bw_i,y0+36),c_dim,-1)
            cv2.rectangle(p4,(bar_x0,y0+20),(bar_x0+bw_i,y0+36),c,1)
        cv2.putText(p4,
            f"{nombre}: D={pd_v*100:.1f}%({cd_v}) I={pi_v*100:.1f}%({ci_v})",
            (bar_x0,y0+row_h-5),cv2.FONT_HERSHEY_SIMPLEX,0.28,(210,210,210),1)

    panels = [r(p) for p in [p1,p2,p3,p4]]
    canvas = np.hstack(panels)
    cab    = np.zeros((46,canvas.shape[1],3),dtype=np.uint8); cab[:]=(40,40,60)
    cv2.putText(cab,
        f"{archivo[:30]}  fondo={agr_f}  pantal.={agr_p}  "
        f"Jaccard={jac_val:.4f}",
        (8,30),cv2.FONT_HERSHEY_SIMPLEX,0.44,(255,255,255),1)
    etq = np.zeros((22,canvas.shape[1],3),dtype=np.uint8); etq[:]=(25,25,40)
    pw  = canvas.shape[1]//4
    for idx,t in enumerate(["1-Original","2-Mascara+Eje",
                             "3-KMeans K=5","4-Histograma"]):
        cv2.putText(etq,t,(idx*pw+4,15),cv2.FONT_HERSHEY_SIMPLEX,0.37,(180,180,180),1)
    final = np.vstack([cab,etq,canvas])
    cv2.imshow(f"f{agr_f}_p{agr_p}_{archivo[:12]}", final)
    cv2.waitKey(1)

# ─────────────────────────────────────────────────────────────────────────────
def seleccionar():
    root=tk.Tk(); root.withdraw(); root.attributes('-topmost',True)
    messagebox.showinfo("Análisis de Sensibilidad v10",
        "Selecciona la carpeta de UN participante\n"
        "de 'imagenes limpias' (salida de Fase 1).\n\n"
        f"Fondo:      {RANGO_FONDO}\n"
        f"Pantaloneta: {RANGO_PANTALONETA}\n\n"
        "Imágenes en tiempo real.\n"
        "Gráficas y Excel al finalizar.")
    ruta=filedialog.askdirectory(title="Selecciona la carpeta")
    root.destroy()
    if not ruta: sys.exit(0)
    return ruta

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n"+"="*65)
    print("  ANÁLISIS DE SENSIBILIDAD v10")
    print(f"  Fondo:       {RANGO_FONDO}")
    print(f"  Pantaloneta: {RANGO_PANTALONETA}")
    print(f"  K-Means fijo: K={K_FIJO}")
    print("="*65)

    ruta=seleccionar()
    print(f"\n[✔] Carpeta: {ruta}")

    imagenes=sorted([f for f in os.listdir(ruta)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))])
    if not imagenes:
        print("[!] Sin imágenes."); input("Enter..."); sys.exit(1)

    total=len(imagenes)*len(RANGO_FONDO)*len(RANGO_PANTALONETA)
    print(f"\n  Imágenes     : {len(imagenes)}")
    print(f"  Combinaciones: {total}")
    print(f"  (fondo × pantaloneta = {len(RANGO_FONDO)} × {len(RANGO_PANTALONETA)})")
    if input("\n  ¿Continuar? (s/n): ").strip().lower()!='s':
        sys.exit(0)

    registros=[]
    contador=0

    for archivo in imagenes:
        img=cv2.imdecode(
            np.fromfile(os.path.join(ruta,archivo),dtype=np.uint8),
            cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [!] No cargó: {archivo}"); continue

        h_img,w_img=img.shape[:2]
        print(f"\n  ══ {archivo}  ({w_img}x{h_img})")

        img_sb, barra_x = eliminar_barra(img)
        print(f"     Barra eliminada: x={barra_x}")

        for agr_f in RANGO_FONDO:
            for agr_p in RANGO_PANTALONETA:
                contador+=1
                pct=(contador/total)*100
                print(f"    [{pct:5.1f}%] fondo={agr_f:2d} pantal={agr_p:3d}",
                      end=" ... ")
                sys.stdout.flush()

                try:
                    # Segmentación dos pasos
                    mask=segmentar_dos_pasos(img_sb,agr_f,agr_p)
                    px_tot=int(np.sum(mask>0))

                    if px_tot < 1000:
                        print(f"OMITIDA — máscara pequeña {px_tot}px")
                        registros.append({
                            "Imagen":archivo,"Agr_Fondo":agr_f,
                            "Agr_Pantaloneta":agr_p,
                            "Jaccard":None,
                            "Detalle":f"Máscara pequeña {px_tot}px"})
                        continue

                    # K-Means
                    img_sem=kmeans_semaforo(img_sb,mask)

                    # Separar piernas
                    md,mi,eje,sep_info=separar_piernas(mask)

                    if md is None:
                        print(f"OMITIDA — {sep_info}")
                        registros.append({
                            "Imagen":archivo,"Agr_Fondo":agr_f,
                            "Agr_Pantaloneta":agr_p,
                            "Jaccard":None,"Detalle":sep_info})
                        mostrar(archivo,agr_f,agr_p,img,img_sb,
                                mask,img_sem,None,None,-1,0.0,
                                {n:0.0 for n in COLORES_ACTIVOS},
                                {n:0.0 for n in COLORES_ACTIVOS},
                                {n:0 for n in COLORES_ACTIVOS},
                                {n:0 for n in COLORES_ACTIVOS},
                                sep_info,barra_x)
                        continue

                    # Histogramas y Jaccard
                    prop_d,cnt_d,tot_d=histograma_color(img_sem,md)
                    prop_i,cnt_i,tot_i=histograma_color(img_sem,mi)

                    if tot_d==0 or tot_i==0:
                        print("OMITIDA — pierna sin colores activos")
                        registros.append({
                            "Imagen":archivo,"Agr_Fondo":agr_f,
                            "Agr_Pantaloneta":agr_p,
                            "Jaccard":None,
                            "Detalle":"Pierna sin colores activos"})
                        continue

                    jac_val=jaccard_histograma(prop_d,prop_i)
                    px_d=int(np.sum(md>0)); px_i=int(np.sum(mi>0))
                    print(f"Jaccard={jac_val:.4f}  der={px_d}px izq={px_i}px")

                    # Mostrar ventana
                    mostrar(archivo,agr_f,agr_p,img,img_sb,
                            mask,img_sem,md,mi,eje,jac_val,
                            prop_d,prop_i,cnt_d,cnt_i,sep_info,barra_x)

                    fila={
                        "Imagen":archivo,
                        "Agr_Fondo":agr_f,
                        "Agr_Pantaloneta":agr_p,
                        "Jaccard":jac_val,
                        "Px_Der":px_d,"Px_Izq":px_i,
                        "Eje_X":eje,"Detalle":sep_info
                    }
                    for n in COLORES_ACTIVOS:
                        fila[f"der_{n}_pct"]=round(prop_d[n]*100,2)
                        fila[f"izq_{n}_pct"]=round(prop_i[n]*100,2)
                        fila[f"der_{n}_px"] =cnt_d[n]
                        fila[f"izq_{n}_px"] =cnt_i[n]
                    registros.append(fila)

                except Exception as e:
                    import traceback
                    print(f"ERROR: {e}"); traceback.print_exc()
                    registros.append({
                        "Imagen":archivo,"Agr_Fondo":agr_f,
                        "Agr_Pantaloneta":agr_p,
                        "Jaccard":None,"Detalle":str(e)})

    # ── Excel ──────────────────────────────────────────────────────────────────
    if not registros:
        print("[!] Sin resultados.")
        cv2.destroyAllWindows(); input("Enter..."); sys.exit(1)

    df  =pd.DataFrame(registros)
    df_v=df[df["Jaccard"].notna()&(df["Jaccard"]>0)]

    ruta_xl=os.path.join(ruta,"analisis_sensibilidad_v10.xlsx")
    with pd.ExcelWriter(ruta_xl,engine='openpyxl') as wr:
        df.to_excel(wr,sheet_name="Datos_Crudos",index=False)
        if len(df_v):
            # Stats por agresividad de fondo (promediando sobre todos los valores de pantaloneta)
            (df_v.groupby("Agr_Fondo")["Jaccard"]
             .agg(Promedio='mean',Mediana='median',
                  Desv_Est='std',Minimo='min',Maximo='max')
             .round(4).reset_index()
             .to_excel(wr,sheet_name="Stats_Fondo",index=False))
            # Stats por agresividad de pantaloneta (promediando sobre todos los valores de fondo)
            (df_v.groupby("Agr_Pantaloneta")["Jaccard"]
             .agg(Promedio='mean',Mediana='median',
                  Desv_Est='std',Minimo='min',Maximo='max')
             .round(4).reset_index()
             .to_excel(wr,sheet_name="Stats_Pantaloneta",index=False))
            # Mejor combinación
            idx_m=df_v["Jaccard"].idxmax(); m=df_v.loc[idx_m]
            pd.DataFrame([{
                "Total_Imagenes"        :len(imagenes),
                "Total_Combinaciones"   :len(registros),
                "Jaccard_Promedio"      :round(df_v['Jaccard'].mean(),4),
                "Jaccard_Mediana"       :round(df_v['Jaccard'].median(),4),
                "Jaccard_Desv_Est"      :round(df_v['Jaccard'].std(),4),
                "Jaccard_Min"           :round(df_v['Jaccard'].min(),4),
                "Jaccard_Max"           :round(df_v['Jaccard'].max(),4),
                "Mejor_Agr_Fondo"       :int(m['Agr_Fondo']),
                "Mejor_Agr_Pantaloneta" :int(m['Agr_Pantaloneta']),
                "Mejor_Jaccard"         :round(m['Jaccard'],4),
                "Valor_Fondo_Sistema"   :VALOR_FONDO_USADO,
                "Valor_Pantal_Sistema"  :VALOR_PANTALONETA_USADO,
            }]).to_excel(wr,sheet_name="Resumen_Global",index=False)

    print(f"\n[✔] Excel guardado: {ruta_xl}")

    # ── DOS GRÁFICAS AL FINAL ──────────────────────────────────────────────────
    if len(df_v):
        # Paletas de colores para las líneas
        # Gráfica 1: una línea por cada valor de pantaloneta
        # Gráfica 2: una línea por cada valor de fondo
        cmap1 = plt.cm.plasma     # para pantaloneta en gráfica 1
        cmap2 = plt.cm.viridis    # para fondo en gráfica 2

        cols_p = cmap1(np.linspace(0.05, 0.90, len(RANGO_PANTALONETA)))
        cols_f = cmap2(np.linspace(0.10, 0.85, len(RANGO_FONDO)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.patch.set_facecolor('#F8F9FA')

        # ── Gráfica 1: X = Agresividad de FONDO, Y = Jaccard
        #    Una línea por cada valor de agresividad de pantaloneta
        ax1.set_facecolor('#F0F4F8')

        mean_f_global = df_v.groupby("Agr_Fondo")["Jaccard"].mean()

        for i, pval in enumerate(RANGO_PANTALONETA):
            sub  = df_v[df_v["Agr_Pantaloneta"] == pval]
            grp  = sub.groupby("Agr_Fondo")["Jaccard"].mean()
            if grp.empty:
                continue
            # Línea resaltada si contiene el valor de pantaloneta usado
            if pval == VALOR_PANTALONETA_USADO:
                ax1.plot(grp.index, grp.values,
                         marker='o', ms=6, lw=2.8,
                         color='#B91C1C', zorder=6,
                         label=f'p={pval} ← usado en sistema')
            else:
                ax1.plot(grp.index, grp.values,
                         marker='o', ms=4, lw=1.4,
                         color=cols_p[i], alpha=0.75,
                         label=f'p={pval}')

        # Línea del promedio global (todas las pantalonetas)
        ax1.plot(mean_f_global.index, mean_f_global.values,
                 marker='D', ms=7, lw=2.5, color='#1F4E79',
                 ls='--', zorder=7, label='Promedio global')

        # Línea vertical en el valor de fondo usado
        ax1.axvline(x=VALOR_FONDO_USADO,
                    color='#1F4E79', ls=':', lw=2,
                    label=f'Fondo usado: {VALOR_FONDO_USADO}')

        # Punto máximo del promedio global
        mejor_f = mean_f_global.idxmax()
        ax1.scatter([mejor_f], [mean_f_global[mejor_f]],
                    color='gold', s=150, zorder=8, edgecolors='#1F4E79', lw=1.5)
        ax1.annotate(f" Mejor fondo={mejor_f}\n Jaccard={mean_f_global[mejor_f]:.3f}",
                     (mejor_f, mean_f_global[mejor_f]),
                     fontsize=8.5, color='#1F4E79', fontweight='bold',
                     xytext=(5, -25), textcoords='offset points')

        ax1.set_xlabel("Agresividad de fondo ambiental", fontsize=12)
        ax1.set_ylabel("Índice de Jaccard promedio", fontsize=12)
        ax1.set_title(
            "Jaccard vs Agresividad de fondo\n"
            "(cada línea = un valor fijo de agresividad de pantaloneta)",
            fontsize=12, fontweight='bold', color='#1F4E79')
        ax1.grid(True, alpha=0.35)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(min(RANGO_FONDO)-3, max(RANGO_FONDO)+3)

        # Leyenda con scroll — fuera del plot para no tapar líneas
        ax1.legend(fontsize=7, loc='upper left',
                   bbox_to_anchor=(1.01, 1), borderaxespad=0,
                   title="Agresividad\npantaloneta", title_fontsize=8,
                   framealpha=0.9)

        # ── Gráfica 2: X = Agresividad de PANTALONETA, Y = Jaccard
        #    Una línea por cada valor de agresividad de fondo
        ax2.set_facecolor('#F0F4F8')

        mean_p_global = df_v.groupby("Agr_Pantaloneta")["Jaccard"].mean()

        for i, fval in enumerate(RANGO_FONDO):
            sub  = df_v[df_v["Agr_Fondo"] == fval]
            grp  = sub.groupby("Agr_Pantaloneta")["Jaccard"].mean()
            if grp.empty:
                continue
            if fval == VALOR_FONDO_USADO:
                ax2.plot(grp.index, grp.values,
                         marker='s', ms=6, lw=2.8,
                         color='#B91C1C', zorder=6,
                         label=f'f={fval} ← usado en sistema')
            else:
                ax2.plot(grp.index, grp.values,
                         marker='s', ms=4, lw=1.4,
                         color=cols_f[i], alpha=0.75,
                         label=f'f={fval}')

        ax2.plot(mean_p_global.index, mean_p_global.values,
                 marker='D', ms=7, lw=2.5, color='#1F4E79',
                 ls='--', zorder=7, label='Promedio global')

        ax2.axvline(x=VALOR_PANTALONETA_USADO,
                    color='#1F4E79', ls=':', lw=2,
                    label=f'Pantaloneta usada: {VALOR_PANTALONETA_USADO}')

        mejor_p = mean_p_global.idxmax()
        ax2.scatter([mejor_p], [mean_p_global[mejor_p]],
                    color='gold', s=150, zorder=8, edgecolors='#1F4E79', lw=1.5)
        ax2.annotate(
            f" Mejor pantal.={mejor_p}\n Jaccard={mean_p_global[mejor_p]:.3f}",
            (mejor_p, mean_p_global[mejor_p]),
            fontsize=8.5, color='#1F4E79', fontweight='bold',
            xytext=(5, -25), textcoords='offset points')

        ax2.set_xlabel("Agresividad de pantaloneta deportiva", fontsize=12)
        ax2.set_ylabel("Índice de Jaccard promedio", fontsize=12)
        ax2.set_title(
            "Jaccard vs Agresividad de pantaloneta\n"
            "(cada línea = un valor fijo de agresividad de fondo)",
            fontsize=12, fontweight='bold', color='#1F4E79')
        ax2.grid(True, alpha=0.35)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(min(RANGO_PANTALONETA)-3, max(RANGO_PANTALONETA)+3)

        ax2.legend(fontsize=7, loc='upper left',
                   bbox_to_anchor=(1.01, 1), borderaxespad=0,
                   title="Agresividad\nfondo", title_fontsize=8,
                   framealpha=0.9)

        plt.suptitle(
            "Análisis de sensibilidad: Índice de Jaccard (histograma de color) "
            "vs parámetros de agresividad\n"
            f"K-Means fijo K={K_FIJO} | azul excluido | "
            f"línea roja = valor usado en el sistema | "
            f"estrella dorada = mejor valor empírico",
            fontsize=10, color='#374151', y=1.01)

        plt.tight_layout()

        ruta_g = os.path.join(ruta, "Fig7_sensibilidad_agresividad.png")
        plt.savefig(ruta_g, dpi=180, bbox_inches='tight', facecolor='#F8F9FA')
        plt.close()
        print(f"[✔] Gráfica guardada: {ruta_g}")

        print(f"\n  ── ESTADÍSTICAS FINALES ──")
        print(f"  Jaccard promedio global : {df_v['Jaccard'].mean():.4f}")
        print(f"  Mediana                 : {df_v['Jaccard'].median():.4f}")
        print(f"  Desviación estándar     : {df_v['Jaccard'].std():.4f}")
        print(f"  Mínimo                  : {df_v['Jaccard'].min():.4f}")
        print(f"  Máximo                  : {df_v['Jaccard'].max():.4f}")
        mejor_f  = mean_f.idxmax()
        mejor_p  = mean_p.idxmax()
        print(f"\n  Mejor agresividad de fondo       : {mejor_f}"
              f" (Jaccard={mean_f[mejor_f]:.4f})")
        print(f"  Mejor agresividad de pantaloneta : {mejor_p}"
              f" (Jaccard={mean_p[mejor_p]:.4f})")
        print(f"\n  Valor usado en sistema — fondo      : {VALOR_FONDO_USADO}"
              f" → Jaccard={mean_f.get(VALOR_FONDO_USADO, float('nan')):.4f}")
        print(f"  Valor usado en sistema — pantaloneta: {VALOR_PANTALONETA_USADO}"
              f" → Jaccard={mean_p.get(VALOR_PANTALONETA_USADO, float('nan')):.4f}")

    print("\n[✔] Análisis completo.")
    print("    Las ventanas siguen abiertas. Presiona una tecla para cerrarlas.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    input("\nPresiona Enter para cerrar el programa...")

if __name__=="__main__":
    main()