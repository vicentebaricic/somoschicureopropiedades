"""Normaliza brillo, contraste y balance de color de los retratos del equipo.

Calcula la media y la desviación estándar de cada canal (R, G, B) en las tres
fotos y lleva cada una al promedio común, de modo que luz y tono queden
parejos antes de aplicar el filtro CSS del sitio. Además recorta a 3:4 y
redimensiona para que las tres tengan el mismo tamaño.

Uso (desde la raíz del proyecto):
    pip install Pillow
    python scripts/normalizar_fotos.py                 # guarda en assets/agentes/normalizadas/
    python scripts/normalizar_fotos.py --reemplazar    # sobrescribe los originales (deja copia .orig.jpg)
    python scripts/normalizar_fotos.py --fuerza 0.6    # corrección parcial (0 = nada, 1 = total)
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageOps, ImageStat

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "assets" / "agentes"
ARCHIVOS = ["corredora1.jpg", "corredora2.jpg", "corredora3.jpg"]
TAMANO = (900, 1200)  # 3:4, igual al aspect-ratio del sitio


def estadisticas(img):
    # Solo el tercio central, donde está la persona, para que el fondo no domine.
    w, h = img.size
    zona = img.crop((w // 6, h // 8, w - w // 6, h - h // 8))
    stat = ImageStat.Stat(zona)
    return stat.mean, [max(s, 1.0) for s in stat.stddev]


def ajustar(img, media, desv, media_obj, desv_obj, fuerza):
    canales = []
    for banda, m, s, mo, so in zip(img.split(), media, desv, media_obj, desv_obj):
        m_final = m + (mo - m) * fuerza
        s_final = s + (so - s) * fuerza
        escala = s_final / s
        lut = [max(0, min(255, round((v - m) * escala + m_final))) for v in range(256)]
        canales.append(banda.point(lut))
    return Image.merge("RGB", canales)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reemplazar", action="store_true", help="sobrescribe los originales")
    parser.add_argument("--fuerza", type=float, default=1.0, help="intensidad de la corrección, 0 a 1")
    args = parser.parse_args()

    rutas = [CARPETA / nombre for nombre in ARCHIVOS]
    faltan = [r.name for r in rutas if not r.exists()]
    if faltan:
        raise SystemExit(f"Faltan fotos en {CARPETA}: {', '.join(faltan)}")

    fotos = []
    for ruta in rutas:
        img = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
        img = ImageOps.fit(img, TAMANO, Image.LANCZOS, centering=(0.5, 0.35))
        fotos.append((ruta, img, *estadisticas(img)))

    media_obj = [sum(f[2][c] for f in fotos) / len(fotos) for c in range(3)]
    desv_obj = [sum(f[3][c] for f in fotos) / len(fotos) for c in range(3)]
    print("Objetivo RGB  media:", [round(v, 1) for v in media_obj], " contraste:", [round(v, 1) for v in desv_obj])

    destino = CARPETA if args.reemplazar else CARPETA / "normalizadas"
    destino.mkdir(exist_ok=True)
    for ruta, img, media, desv in fotos:
        salida = ajustar(img, media, desv, media_obj, desv_obj, max(0.0, min(1.0, args.fuerza)))
        if args.reemplazar:
            shutil.copy2(ruta, ruta.with_suffix(".orig.jpg"))
        salida.save(destino / ruta.name, quality=88, optimize=True, progressive=True)
        print(f"{ruta.name}: media {[round(v, 1) for v in media]} -> {destino / ruta.name}")


if __name__ == "__main__":
    main()
