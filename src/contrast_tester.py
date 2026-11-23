from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path

CONTRASTS = [1.0]

def generate_contrast_images(pathofimage, output_folder):
    original = Image.open(pathofimage).convert("RGB")
    grayscale = ImageOps.grayscale(original)
    enhancement = ImageEnhance.Contrast(grayscale)

    generated_images = []

    for contrast in CONTRASTS:
        image_p = enhancement.enhance(contrast)
        pathofsave = Path(output_folder) / f"{pathofimage.stem}_{contrast}.png"
        image_p.save(pathofsave)
        generated_images.append((contrast, pathofsave))

        print(f"{pathofimage.stem} {contrast} done")

    return generated_images
