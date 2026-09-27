"""Empareja los retratos del equipo para que se vean como una misma sesión.

Qué hace, en orden:
  1. Encuadre: detecta la cara (si OpenCV está instalado) y recorta a 3:4 para
     que las tres caras queden al mismo tamaño y a la misma altura.
  2. Luz: lleva el brillo medio de la cara y el contraste general al promedio
     de las tres fotos.
  3. Color: acerca el balance de blancos de la piel al promedio (sin
     uniformar del todo los tonos de piel) y deja la saturación pareja,
     algo más baja, para que los fondos no compitan.

Uso (desde la raíz del proyecto):
    pip install Pillow "opencv-python-headless<5"   # OpenCV es opcional
    python scripts/normalizar_fotos.py                  # guarda en assets/agentes/normalizadas/
    python scripts/normalizar_fotos.py --reemplazar     # sobrescribe (deja copia .orig.jpg)
    python scripts/normalizar_fotos.py --origen carpeta/con/originales
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps, ImageStat

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "assets" / "agentes"
ARCHIVOS = ["corredora1.jpg", "corredora2.jpg", "corredora3.jpg"]
TAMANO = (900, 1200)      # 3:4, igual al aspect-ratio del sitio
CARA_Y = 0.36             # altura del centro de la cara dentro del recorte
FUERZA_COLOR = 0.6        # 0 = respeta el tono original, 1 = iguala la piel por completo
SATURACION = 0.9          # saturación final relativa al promedio de las tres


def detectar_cara(img):
    """Devuelve (x, y, ancho, alto) de la cara más grande, o None."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if not hasattr(cv2, "CascadeClassifier"):  # OpenCV 5 ya no trae los clasificadores Haar
        return None
    gris = np.array(img.convert("L"))
    escala = 1200 / max(gris.shape)
    chica = cv2.resize(gris, None, fx=escala, fy=escala) if escala < 1 else gris
    escala = min(escala, 1)
    clasif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    caras, _, confianza = clasif.detectMultiScale3(
        chica, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40), outputRejectLevels=True
    )
    if len(caras) == 0:
        return None
    x, y, w, h = caras[int(np.argmax(confianza))]
    return tuple(int(v / escala) for v in (x, y, w, h))


def fraccion_minima(img, cara):
    """Menor proporción ancho-cara / ancho-recorte que admite la foto."""
    W, H = img.size
    ancho_max = min(W, H * 3 / 4)
    return cara[2] / ancho_max


def recortar(img, cara, fraccion):
    W, H = img.size
    x, y, w, h = cara
    ancho = min(w / fraccion, W, H * 3 / 4)
    alto = ancho * 4 / 3
    cx, cy = x + w / 2, y + h / 2
    izq = min(max(cx - ancho / 2, 0), W - ancho)
    arr = min(max(cy - alto * CARA_Y, 0), H - alto)
    caja = (round(izq), round(arr), round(izq + ancho), round(arr + alto))
    escala = TAMANO[0] / ancho
    cara_nueva = tuple(round(v) for v in ((x - izq) * escala, (y - arr) * escala, w * escala, h * escala))
    return img.crop(caja).resize(TAMANO, Image.LANCZOS), cara_nueva


def zona_piel(cara):
    # Centro de la cara (mejillas, nariz), evitando pelo y fondo.
    x, y, w, h = cara
    return (x + w * 0.25, y + h * 0.35, x + w * 0.75, y + h * 0.8)


def saturacion_media(img):
    return ImageStat.Stat(img.convert("HSV").split()[1]).mean[0]


def aplicar_curva(img, ganancias, desplazamiento, contraste, media_l):
    canales = []
    for banda, g in zip(img.split(), ganancias):
        lut = [max(0, min(255, round(((v * g + desplazamiento) - media_l) * contraste + media_l))) for v in range(256)]
        canales.append(banda.point(lut))
    return Image.merge("RGB", canales)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", type=Path, default=CARPETA, help="carpeta con las fotos originales")
    parser.add_argument("--reemplazar", action="store_true", help="sobrescribe las fotos en assets/agentes")
    args = parser.parse_args()

    rutas = [args.origen / nombre for nombre in ARCHIVOS]
    faltan = [r.name for r in rutas if not r.exists()]
    if faltan:
        raise SystemExit(f"Faltan fotos en {args.origen}: {', '.join(faltan)}")

    # 1. Encuadre común
    fotos = []
    for ruta in rutas:
        img = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
        cara = detectar_cara(img)
        if cara is None:
            W, H = img.size
            print(f"{ruta.name}: no se detectó cara, se usa el centro")
            cara = (W * 0.35, H * 0.2, W * 0.3, W * 0.3)
        fotos.append([ruta, img, cara])
    fraccion = max(fraccion_minima(img, cara) for _, img, cara in fotos)
    for f in fotos:
        f[1], f[2] = recortar(f[1], f[2], fraccion)
    print(f"Cara ocupa el {fraccion:.0%} del ancho en las tres fotos")

    # 2 y 3. Luz y color
    piel = [ImageStat.Stat(img.crop(zona_piel(cara))).mean for _, img, cara in fotos]
    luz_piel = [sum(p) / 3 for p in piel]
    contrastes = [ImageStat.Stat(img.convert("L")).stddev[0] for _, img, _ in fotos]
    obj_piel = [sum(p[c] for p in piel) / len(piel) for c in range(3)]
    obj_luz = sum(luz_piel) / len(luz_piel)
    obj_contraste = sum(contrastes) / len(contrastes)

    corregidas = []
    for (ruta, img, cara), p, luz, contraste in zip(fotos, piel, luz_piel, contrastes):
        # Balance de blancos parcial hacia la piel promedio, y brillo de piel al promedio.
        ganancias = []
        for c in range(3):
            tono_obj = p[c] + (obj_piel[c] * luz / obj_luz - p[c]) * FUERZA_COLOR
            ganancias.append(tono_obj / max(p[c], 1) * obj_luz / luz)
        media_l = obj_luz
        img = aplicar_curva(img, ganancias, 0, obj_contraste / max(contraste, 1), media_l)
        corregidas.append([ruta, img])

    sats = [saturacion_media(img) for _, img in corregidas]
    obj_sat = sum(sats) / len(sats) * SATURACION
    destino = CARPETA if args.reemplazar else CARPETA / "normalizadas"
    destino.mkdir(exist_ok=True)
    for (ruta, img), sat in zip(corregidas, sats):
        img = ImageEnhance.Color(img).enhance(obj_sat / max(sat, 1))
        salida = destino / ruta.name
        if args.reemplazar and salida.exists():
            shutil.copy2(salida, salida.with_suffix(".orig.jpg"))
        img.save(salida, quality=86, optimize=True, progressive=True)
        print(f"{ruta.name} -> {salida}")


if __name__ == "__main__":
    main()
