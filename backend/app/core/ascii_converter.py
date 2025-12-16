import numpy as np
from PIL import Image


class ASCIIConverter:
    # Conjunto de caracteres Unicode ordenados por densidad
    UNICODE_CHARS = [
        '⠀',  # Espacio en blanco (más claro)
        '⣀', '⣁', '⣂', '⣃', '⣄', '⣅', '⣆', '⣇', '⣈', '⣉', '⣊', '⣋', '⣌', '⣍', '⣎', '⣏',
        '⣐', '⣑', '⣒', '⣓', '⣔', '⣕', '⣖', '⣗', '⣘', '⣙', '⣚', '⣛', '⣜', '⣝', '⣞', '⣟',
        '⣠', '⣡', '⣢', '⣣', '⣤', '⣥', '⣦', '⣧', '⣨', '⣩', '⣪', '⣫', '⣬', '⣭', '⣮', '⣯',
        '⣰', '⣱', '⣲', '⣳', '⣴', '⣵', '⣶', '⣷', '⣸', '⣹', '⣺', '⣻', '⣼', '⣽', '⣾', '⣿'
    ]

    @staticmethod
    def floyd_steinberg_dithering(image):
        """Aplica el algoritmo de Floyd-Steinberg Dithering a una imagen en escala de grises."""
        pixels = np.array(image, dtype=float)
        height, width = pixels.shape
        num_levels = len(ASCIIConverter.UNICODE_CHARS) - 1

        for y in range(height):
            for x in range(width):
                old_pixel = pixels[y, x]

                # Cuantizar el píxel al nivel más cercano
                new_pixel = round(old_pixel / 255.0 * num_levels)
                new_pixel = max(0, min(num_levels, new_pixel))  # Clamp

                pixels[y, x] = new_pixel

                # Calcular el error de cuantización
                quant_error = old_pixel - (new_pixel / num_levels * 255.0)

                # Distribuir el error a los píxeles vecinos
                if x + 1 < width:
                    pixels[y, x + 1] += quant_error * 7 / 16
                if y + 1 < height:
                    if x > 0:
                        pixels[y + 1, x - 1] += quant_error * 3 / 16
                    pixels[y + 1, x] += quant_error * 5 / 16
                    if x + 1 < width:
                        pixels[y + 1, x + 1] += quant_error * 1 / 16

        return pixels.astype(int)

    @staticmethod
    def image_to_ascii(image, max_width=100):
        """Convierte una imagen a arte ASCII usando caracteres Unicode."""
        # Convertir a escala de grises
        gray_image = image.convert('L')

        # Calcular nueva altura manteniendo proporción
        aspect_ratio = gray_image.height / gray_image.width
        new_height = int(max_width * aspect_ratio * 0.55)  # 0.55 corrige la proporción de los caracteres

        # Redimensionar imagen
        resized_image = gray_image.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # Aplicar dithering
        dithered = ASCIIConverter.floyd_steinberg_dithering(resized_image)

        # Convertir a ASCII
        ascii_art = []
        for row in dithered:
            line = ''.join([ASCIIConverter.UNICODE_CHARS[pixel] for pixel in row])
            ascii_art.append(line)

        return '\n'.join(ascii_art)