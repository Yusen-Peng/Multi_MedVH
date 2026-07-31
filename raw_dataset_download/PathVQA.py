import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET_NAME = "flaviagiammarino/path-vqa"
SPLIT = "test"

NUM_SAMPLES = 1000
RANDOM_SEED = 42

OUTPUT_ROOT = Path("path_vqa_test_1000")
IMAGE_DIR = OUTPUT_ROOT / "images"
JSON_PATH = OUTPUT_ROOT / "annotations.json"

OUTPUT_EXTENSION = ".jpg"


# -------------------------------------------------------------------------
# Image saving
# -------------------------------------------------------------------------

def save_image(image: Image.Image, output_path: Path) -> None:
    """
    Save a PIL image as JPEG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = image.convert("RGB")
    image.save(
        output_path,
        format="JPEG",
        quality=95,
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_NAME}, split={SPLIT}...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT,
    )

    # Randomly shuffle the test set and keep only 1,000 examples.
    sample_size = min(NUM_SAMPLES, len(dataset))

    dataset = (
        dataset
        .shuffle(seed=RANDOM_SEED)
        .select(range(sample_size))
    )

    print(
        f"Selected {len(dataset)} examples "
        f"using random seed {RANDOM_SEED}."
    )

    annotations = []
    failed_rows = []

    for row_index, row in enumerate(
        tqdm(dataset, desc="Exporting PathVQA")
    ):
        try:
            image = row["image"]
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()

            if image is None:
                raise ValueError("The image field is empty.")

            if not question:
                raise ValueError("The question field is empty.")

            if not answer:
                raise ValueError("The answer field is empty.")

            # Fake but deterministic ID within this sampled dataset.
            question_id = f"pathvqa-{SPLIT}-{row_index:06d}"

            image_filename = f"{question_id}{OUTPUT_EXTENSION}"
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
                    "category": "pathology",
                }
            )

        except Exception as error:
            failed_rows.append(
                {
                    "row_index": row_index,
                    "error": str(error),
                }
            )

            print(
                f"\nWarning: failed to process row "
                f"{row_index}: {error}"
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
    print(f"Requested:   {sample_size} entries")
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
