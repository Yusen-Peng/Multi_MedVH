import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET_NAME = "MM-Hallu/HEAL-MedVQA"
CONFIG_NAME = "test"
SPLIT = "test"

NUM_SAMPLES = 1000
RANDOM_SEED = 42

OUTPUT_ROOT = Path("cxr_heal_medvqa_test_1000")
IMAGE_DIR = OUTPUT_ROOT / "images"
JSON_PATH = OUTPUT_ROOT / "annotations.json"

OUTPUT_EXTENSION = ".jpg"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def sanitize_filename(value: Any) -> str:
    """
    Convert an arbitrary ID into a safe filename.
    """
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "unknown"


def save_image(image: Image.Image, output_path: Path) -> None:
    """
    Save a PIL image as JPEG.

    Chest X-rays are often grayscale, but converting to RGB ensures
    compatibility with JPEG and downstream vision-language pipelines.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image.convert("RGB").save(
        output_path,
        format="JPEG",
        quality=95,
    )


# -------------------------------------------------------------------------
# Main
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

    print(f"Original dataset size: {len(dataset)}")

    sample_size = min(NUM_SAMPLES, len(dataset))

    # Reproducible random downsampling.
    dataset = (
        dataset
        .shuffle(seed=RANDOM_SEED)
        .select(range(sample_size))
    )

    print(
        f"Selected {len(dataset)} questions "
        f"using random seed {RANDOM_SEED}."
    )

    annotations = []
    failed_rows = []

    for sample_index, row in enumerate(
        tqdm(dataset, desc="Exporting HEAL-MedVQA")
    ):
        try:
            image = row["image"]
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            question_id = str(row["question_id"]).strip()
            image_id = str(row["image_id"]).strip()

            if image is None:
                raise ValueError("The image field is empty.")

            if not question:
                raise ValueError("The question field is empty.")

            if not answer:
                raise ValueError("The answer field is empty.")

            if not question_id:
                raise ValueError("The question_id field is empty.")

            # Use both the sample index and image ID to avoid filename
            # collisions when one image has multiple questions.
            safe_image_id = sanitize_filename(image_id)

            image_filename = (
                f"{sample_index:06d}_{safe_image_id}"
                f"{OUTPUT_EXTENSION}"
            )

            image_output_path = IMAGE_DIR / image_filename

            save_image(image, image_output_path)

            relative_image_path = image_output_path.relative_to(
                OUTPUT_ROOT
            )

            annotations.append(
                {
                    "question_id": question_id,
                    "image path": relative_image_path.as_posix(),
                    "prompt": question,
                    "GT": answer,
                    "category": "CXR",
                }
            )

        except Exception as error:
            failed_rows.append(
                {
                    "sample_index": sample_index,
                    "question_id": row.get("question_id"),
                    "image_id": row.get("image_id"),
                    "error": str(error),
                }
            )

            print(
                f"\nWarning: failed to process sample "
                f"{sample_index}: {error}"
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
    print(f"Requested:   {sample_size} questions")
    print(f"Exported:    {len(annotations)} questions")

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