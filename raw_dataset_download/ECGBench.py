import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm




DATASET_NAME = "PULSE-ECG/ECGBench"
CONFIG_NAME = "code15-test"
SPLIT = "test"

OUTPUT_ROOT = Path("ecgbench_code15")
IMAGE_DIR = OUTPUT_ROOT / "images"
JSON_PATH = OUTPUT_ROOT / "annotations.json"

# Preserve the original PNG files when possible.
# Set this to ".jpg" if you specifically need JPEG images.
OUTPUT_EXTENSION = ".png"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def sanitize_filename(value: str) -> str:
    """
    Convert an arbitrary dataset ID into a safe filename.
    """
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "unknown"


def extract_question_and_answer(
    conversations: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Extract the human prompt and GPT ground-truth answer.

    Expected format:
    [
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
    ]
    """
    question = None
    answer = None

    for message in conversations:
        role = str(message.get("from", "")).strip().lower()
        value = str(message.get("value", "")).strip()

        if role in {"human", "user"} and question is None:
            question = value
        elif role in {"gpt", "assistant"} and answer is None:
            answer = value

    if question is None:
        raise ValueError(
            f"Could not find a human question in conversations: {conversations}"
        )

    if answer is None:
        raise ValueError(
            f"Could not find a GPT answer in conversations: {conversations}"
        )

    return question, answer


def save_image(image: Image.Image, output_path: Path) -> None:
    """
    Save a Hugging Face PIL image to disk.

    JPEG cannot store alpha/transparency, so images are converted to RGB
    when OUTPUT_EXTENSION is .jpg or .jpeg.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extension = output_path.suffix.lower()

    if extension in {".jpg", ".jpeg"}:
        image.convert("RGB").save(
            output_path,
            format="JPEG",
            quality=95,
        )
    elif extension == ".png":
        image.save(output_path, format="PNG")
    else:
        image.save(output_path)


# -------------------------------------------------------------------------
# Main conversion
# -------------------------------------------------------------------------

def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {DATASET_NAME}, "
        f"config={CONFIG_NAME}, split={SPLIT}..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        CONFIG_NAME,
        split=SPLIT,
    )

    annotations = []
    failed_rows = []

    for row_index, row in enumerate(
        tqdm(dataset, desc="Exporting ECGBench")
    ):
        try:
            dataset_id = str(row["id"])
            question, ground_truth = extract_question_and_answer(
                row["conversations"]
            )

            image = row["image"]

            if image is None:
                raise ValueError("The image field is empty.")

            # Including row_index avoids accidental collisions if IDs repeat.
            safe_id = sanitize_filename(dataset_id)
            image_filename = (
                f"{row_index:06d}_{safe_id}{OUTPUT_EXTENSION}"
            )
            image_output_path = IMAGE_DIR / image_filename

            save_image(image, image_output_path)

            # Store a portable relative path inside the JSON.
            relative_image_path = image_output_path.relative_to(
                OUTPUT_ROOT
            )

            annotations.append(
                {
                    "question_id": dataset_id,
                    "image path": relative_image_path.as_posix(),
                    "prompt": question,
                    "GT": ground_truth,
                    "category": "ECG",
                }
            )

        except Exception as error:
            failed_rows.append(
                {
                    "row_index": row_index,
                    "id": row.get("id"),
                    "error": str(error),
                }
            )
            print(
                f"\nWarning: failed to process row {row_index}: {error}"
            )

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            annotations,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDone ✅")
    print(f"Images:      {IMAGE_DIR.resolve()}")
    print(f"Annotations: {JSON_PATH.resolve()}")
    print(f"Exported:    {len(annotations)} entries")

    if failed_rows:
        failed_path = OUTPUT_ROOT / "failed_rows.json"

        with failed_path.open("w", encoding="utf-8") as file:
            json.dump(
                failed_rows,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Failed:      {len(failed_rows)} rows")
        print(f"Failure log: {failed_path.resolve()}")


if __name__ == "__main__":
    main()