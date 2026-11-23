from pathlib import Path
from main import main as recognition_of_image
from start import recognize_page
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from contrast_tester import generate_contrast_images, CONTRASTS
from tempfile import TemporaryDirectory
from statistics import mean
import string


def levenshtein_distance(correct, predicted):
    rows = len(correct) + 1
    cols = len(predicted) + 1

    distances = []

    for space in range(rows):
        distances.append([0] * cols)

    for i in range(rows):
        distances[i][0] = i

    for j in range(cols):
        distances[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if correct[i - 1] == predicted[j - 1]:
                cost = 0
            else:
                cost = 1

            distances[i][j] = min(
                distances[i - 1][j] + 1,
                distances[i][j - 1] + 1,
                distances[i - 1][j - 1] + cost
            )

    return distances[-1][-1]


def calc_cer(correct, predicted):
    if len(correct) == 0:
        if len(predicted) == 0:
            return 0.0  
        else:
            return float("inf")

    errors = levenshtein_distance(correct, predicted)
    return errors / len(correct)

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def main():
    source_folder = Path(__file__).resolve().parent.parent / "Source"

    correct_texts = {
        "sample1_cropped.png": "Apple creampie",
        "sample2.png": "community center",
        "sample3.png": "hello world",
        "newsample1.png": "Recognize the following text",
        "newsample2.png": "Force equals to the product of mass and acceleration",
        "newsample3.png": "Testing lines 1",
        "newsample4.png": "This teacher sucks",
        "newsample5.png": "I love physics"
    }

    MODEL_NAME = "microsoft/trocr-base-handwritten"

    print("Loading OCR model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.eval()

    results = {}

    for contrast in CONTRASTS:
        results[contrast] = []


    print("\n========== Single line texts ==========")


    with TemporaryDirectory(prefix="ocr_test_", dir=source_folder) as temp_folder:

        for filename, correct in correct_texts.items():

            original_path = source_folder / filename

            if not original_path.exists():
                print(f"Missing original: {original_path}")
                continue

            # Use the same indentation as generated_images below.
            print(f"\n--- {filename} ---")
            generated_images = generate_contrast_images(
                original_path, temp_folder
            )

            for contrast, image_path in generated_images:

                predicted = recognition_of_image(
                    image_path, processor, model
                )

                if predicted is None:
                    continue

                strict_cer = calc_cer(correct, predicted)

                normalized_cer = calc_cer(
                    normalize_text(correct),
                    normalize_text(predicted)
                )

                results[contrast].append(
                    (strict_cer, normalized_cer)
                )

                print(
                    f"{filename}, contrast {contrast}: "
                    f"Strict CER = {strict_cer:.2%}, "
                    f"Normalized CER = {normalized_cer:.2%}"
                )

    print("\n--- Single-line average CER ---")

    for contrast, values in results.items():
        if not values:
            continue

        avg_strict = mean(result[0] for result in values)
        avg_normalized = mean(result[1] for result in values)

        print(
            f"Contrast {contrast}: "
            f"Strict = {avg_strict:.2%}, "
            f"Normalized = {avg_normalized:.2%}"
        )

    print("\nTemporary images have been removed.")

    print("\n========== Multi-line / full page ==========")

    expected_page = "\n".join([
            "My journal",
            "Today is our first day of experimenting with full page",
            "recognitions in our project.",
            "If everything goes right , this page I am extremely writing will",
            "be the first page ever.",
            "Notes for myself , we should definitely implement paragraph",
            "recognition later.",
            "It is quite hard to think of random sentences that",
            "could fill an entire page , I am almost out of ideas.",
            "Leaving bigger space so the page gets filled sooner.",
            "Hello",
            "The project should be able to recognise all kinds",
            "of texts.",
            "What if one sentence is written",
            "with larger font than the other ones?",
        ])

    expected_page_2 = "\n".join([
            "Hello, my recognizer.",
            "This time I will write everything in",
            "the middle of the page.",
            "Keeping lines further apart, and trying",
            "to keep a straight line while writing.",
            "We will see if this hypothetical ideal",
            "situation will improve accurate.",
            "I made a grammatical error in the",
            "last line, I meant to write accuracy.",
            "GPT, you are amazing!",
            "Thus reminds me to test a full",
            "capital letter line.",
        ])
    
    pages = [
        ("full_page_text1.png", expected_page),
        ("full_page_text2.png", expected_page_2),
    ]

    for filename, reference in pages:
        page_path = source_folder / filename

        if not page_path.is_file():
            print(f"Missing page sample: {page_path}")
            continue

        predicted_page = recognize_page(page_path, processor, model)
        if predicted_page is None:
            print(f"{filename}: no text lines detected.")
            continue

        strict_cer = calc_cer(reference, predicted_page)
        normalized_cer = calc_cer(
            normalize_text(reference),
            normalize_text(predicted_page),
        )

        print(f"{filename}: Strict CER = {strict_cer:.2%}, "
              f"Normalized CER = {normalized_cer:.2%}")

if __name__ == "__main__":
    main()
