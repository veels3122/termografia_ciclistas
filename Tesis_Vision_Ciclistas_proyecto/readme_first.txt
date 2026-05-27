================================================================================
GUÍA DE INSTALACIÓN Y EJECUCIÓN: PIPELINE DE TERMOGRAFÍA AUTOMATIZADA
================================================================================
Desarrollado por: Michael Mateo Melgarejo Uribe
Instituciones: Instituto Tecnológico de Roque / Universidad de Cundinamarca
Proyecto: Sistema de Visión Artificial para Caracterización Termográfica
================================================================================

Este documento contiene las instrucciones obligatorias para ejecutar el software 
de procesamiento de imágenes térmicas sin errores de dependencias. Lea 
detenidamente antes de iniciar.

--------------------------------------------------------------------------------
PASO 1: INSTALACIÓN DEL MOTOR OCR (REQUISITO CRÍTICO)
--------------------------------------------------------------------------------
El software utiliza un motor de Reconocimiento Óptico de Caracteres (OCR) para 
extraer las temperaturas máximas y mínimas directamente de las imágenes (Fase 1). 
Python no puede instalar este motor por sí solo.

1. Dentro de esta carpeta comprimida, busque el archivo instalador llamado:
   "tesseract-ocr-setup.exe" (o similar).
2. Ejecútelo como Administrador.
3. IMPORTANTE: Durante la instalación, deje la ruta por defecto intacta. El 
   código buscará el ejecutable estrictamente en la siguiente ruta:
   C:\Program Files\Tesseract-OCR\tesseract.exe

Si omite este paso, la Fase 1 del programa fallará y no podrá leer las temperaturas.

--------------------------------------------------------------------------------
PASO 2: EL PREPROCESADOR Y AUTO-INSTALADOR DE PYTHON
--------------------------------------------------------------------------------
No necesita instalar las librerías matemáticas o de visión artificial manualmente. 
El sistema cuenta con un Motor de Preprocesamiento automatizado.

1. Abra su terminal o consola de comandos (CMD / PowerShell).
2. Navegue hasta la carpeta donde descomprimió este proyecto.
3. Ejecute el script principal (ej. `python main.py` o el nombre de su archivo maestro).

Al arrancar, el software escaneará automáticamente su computadora. Si detecta 
que le faltan librerías requeridas (como opencv-python, pandas, reportlab, scipy 
o pytesseract), la pantalla negra las descargará e instalará en segundo plano 
de forma silenciosa. 
* Nota: Asegúrese de tener conexión a Internet activa en la primera ejecución.

--------------------------------------------------------------------------------
PASO 3: ARQUITECTURA DE DIRECTORIOS (CÓMO USAR EL SISTEMA)
--------------------------------------------------------------------------------
El software procesa los datos en cascada a través de diferentes carpetas. Para 
analizar nuevos atletas, respete la siguiente estructura:

- /imagenes fase 1: Coloque aquí las carpetas de los participantes con las 
  imágenes térmicas originales exportadas por la cámara.
- /data/resultados_termografia_fase1.xlsx: Aquí se guardará y consolidará 
  toda la base de datos maestra con la información geométrica, rangos y simetría.
- /reporte final: Una vez concluida la Fase 5, el sistema generará 
  automáticamente una carpeta por participante aquí, conteniendo el documento 
  PDF con el Reporte Técnico de Caracterización Termográfica.

--------------------------------------------------------------------------------
NOTA TÉCNICA Y ALCANCE DEL SOFTWARE
--------------------------------------------------------------------------------
Este pipeline es un instrumento de medición geométrica y térmica basado en 
visión artificial. Extrae métricas como el Índice de Jaccard (asimetría estructural) 
y el mapeo topográfico de múltiples focos de calor. Los reportes PDF generados 
entregan caracterizaciones matemáticas que deben ser validadas e interpretadas 
por el especialista biomédico o de maestría correspondiente para cualquier 
conclusión clínica final.

================================================================================
Fin del documento.
================================================================================