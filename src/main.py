from pathlib import Path
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

from note_analyzer import note_analysis

# Project settings
MODEL_NAME = "microsoft/trocr-base-handwritten"

def recognize_line(image, processor, model):
    image = image.convert("L").convert("RGB")
    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4
        )

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]


def main(IMAGE_PATH, processor, model):
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}")
        return

    with Image.open(IMAGE_PATH) as image:
        recognized_text = recognize_line(image, processor, model)

    analysis = note_analysis(recognized_text)

    header = (
        f"Title: {analysis['title']}\n"
        f"Subject: {analysis['subject']}\n"
        f"Keywords: {', '.join(analysis['keywords'])}\n"
        "\n--- Recognized text ---\n\n"
    )

    output_text = header + recognized_text

    print("\n--- Recognized Text ---")
    print(recognized_text)
    print("-----------------------")
    return recognized_text

if __name__ == "__main__":
    print("Loading handwriting recognition model...")

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.eval()

    source_folder = Path(__file__).resolve().parent.parent / "Source"

    for filename in ("page2_line4.png", "page2_line4_cleaned.png"):
        image_path = source_folder / filename

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        print(f"\n{filename}")
        print(recognize_line(image, processor, model))