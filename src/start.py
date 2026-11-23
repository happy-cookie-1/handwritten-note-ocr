from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from line_splitter import split_lines
from main import MODEL_NAME, recognize_line, main
from note_analyzer import note_analysis

def recognize_page(image_path, processor, model):
    line_images = split_lines(image_path)
    print(f"Detected {len(line_images)} lines.")

    if not line_images:
        return None

    recognized_lines = []

    for number, line_image in enumerate(line_images, start=1):
        text = recognize_line(line_image, processor, model)
        recognized_lines.append(text)
        print(f"Line {number}: {text}")

    return "\n".join(recognized_lines)

def main():
    source_folder = Path(__file__).resolve().parent.parent / "Source"

    filename = input(
        "Image filename inside folder Source (e.g. NLS1.png): "
    ).strip()

    image_path = source_folder / filename

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return

    print("Loading handwriting recognition model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.eval()

    page_text = recognize_page(image_path, processor, model)

    if page_text is None:
        print("No text lines detected.")
        return

    analysis = note_analysis(page_text)

    header = (
        f"Title: {analysis['title']}\n"
        f"Subject: {analysis['subject']}\n"
        f"Keywords: {', '.join(analysis['keywords'])}\n"
        "\n--- Recognized text ---\n\n"
    )

    output_folder = source_folder.parent / "Output"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{image_path.stem}.txt"
    output_path.write_text(header + page_text, encoding="utf-8")

    print(f"Text and analysis saved to: {output_path}")

if __name__ == "__main__":
    main()