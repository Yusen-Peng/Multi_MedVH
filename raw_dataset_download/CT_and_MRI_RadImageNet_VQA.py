import json
import re
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET_NAME = "raidium/RadImageNet-VQA"
CONFIG_NAME = "benchmark"
SPLIT = "test"

NUM_SAMPLES = 1000
RANDOM_SEED = 42

OUTPUT_ROOT = Path("radimagenet_vqa_benchmark_1000")
IMAGE_DIR = OUTPUT_ROOT / "images"
JSON_PATH = OUTPUT_ROOT / "annotations.json"
FAILED_PATH = OUTPUT_ROOT / "failed_rows.json"

OUTPUT_EXTENSION = ".jpg"


# -------------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------------

def sanitize_filename(value: Any) -> str:
    """
    Convert an arbitrary value into a filesystem-safe filename component.
    """
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "unknown"


def normalize_metadata(metadata: Any) -> dict[str, Any]:
    """
    Ensure metadata is represented as a Python dictionary.

    Depending on the dataset serialization/version, metadata may already
    be a dict or may be provided as a JSON string.
    """
    if metadata is None:
        return {}

    if isinstance(metadata, dict):
        return metadata

    if isinstance(metadata, str):
        metadata = metadata.strip()

        if not metadata:
            return {}

        try:
            parsed = json.loads(metadata)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    try:
        return dict(metadata)
    except (TypeError, ValueError):
        return {}


def normalize_choices(choices: Any) -> list[str]:
    """
    Normalize the choices field into a clean list of strings.

    Non-multiple-choice questions may contain None instead of a list.
    """
    if choices is None:
        return []

    if isinstance(choices, str):
        choices = choices.strip()

        if not choices:
            return []

        # Handle a JSON-serialized list if encountered.
        try:
            parsed = json.loads(choices)

            if isinstance(parsed, list):
                choices = parsed
            else:
                return [choices]
        except json.JSONDecodeError:
            return [choices]

    if isinstance(choices, (list, tuple)):
        normalized = []

        for choice in choices:
            if choice is None:
                continue

            text = str(choice).strip()

            if text:
                normalized.append(text)

        return normalized

    return []


def normalize_question_type(question_type: Any) -> str:
    """
    Normalize variations such as 'multiple_choice' and 'multiple-choice'.
    """
    return str(question_type or "").strip().lower().replace("-", "_")


# -------------------------------------------------------------------------
# Question and answer normalization
# -------------------------------------------------------------------------

def format_question(
    question: str,
    choices: list[str],
    question_type: str,
) -> str:
    """
    Append labeled choices to multiple-choice questions.

    Example:

        What is the anatomical region?

        Choices:
        A. hip
        B. brain
        C. spine
        D. ankle foot
    """
    is_multiple_choice = (
        question_type in {"multiple_choice", "mc", "multiplechoice"}
        or bool(choices)
    )

    if not is_multiple_choice or not choices:
        return question

    choice_lines = [
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(choices)
    ]

    return (
        f"{question}\n\n"
        f"Choices:\n"
        + "\n".join(choice_lines)
    )


def resolve_ground_truth(
    raw_answer: Any,
    choices: list[str],
    metadata: dict[str, Any],
) -> str:
    """
    Produce a textual ground-truth answer.

    Resolution priority:
      1. metadata['correct_text'], if present
      2. map an answer letter such as 'D' to choices[3]
      3. preserve the raw answer

    This avoids storing only 'D' as the GT when the actual answer is
    something like 'ankle foot'.
    """
    correct_text = metadata.get("correct_text")

    if correct_text is not None:
        correct_text = str(correct_text).strip()

        if correct_text and correct_text.lower() not in {
            "none",
            "null",
            "nan",
        }:
            return correct_text

    answer = str(raw_answer or "").strip()

    # Accept formats such as "D", "D.", "(D)", or "Option D".
    letter_match = re.fullmatch(
        r"(?:option\s*)?[\(\[]?([A-Za-z])[\)\].:]?",
        answer,
        flags=re.IGNORECASE,
    )

    if letter_match and choices:
        letter = letter_match.group(1).upper()
        choice_index = ord(letter) - ord("A")

        if 0 <= choice_index < len(choices):
            return choices[choice_index]

    return answer


def get_category(metadata: dict[str, Any]) -> str:
    """
    Use the medical imaging modality as the category.

    Expected benchmark modalities are mainly CT and MRI.
    """
    modality = metadata.get("modality")

    if modality is None:
        return "radiology"

    modality = str(modality).strip().upper()

    if not modality or modality in {"NONE", "NULL", "NAN"}:
        return "radiology"

    return modality.upper()


# -------------------------------------------------------------------------
# Image handling
# -------------------------------------------------------------------------

def save_image(image: Image.Image, output_path: Path) -> None:
    """
    Save a PIL image as JPEG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image.convert("RGB").save(
        output_path,
        format="JPEG",
        quality=95,
    )


# -------------------------------------------------------------------------
# Main export
# -------------------------------------------------------------------------

def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {DATASET_NAME}, "
        f"config={CONFIG_NAME}, split={SPLIT}..."
    )

    # The repository is gated.
    # Run `huggingface-cli login` first, or pass token=YOUR_HF_TOKEN.
    dataset = load_dataset(
        DATASET_NAME,
        CONFIG_NAME,
        split=SPLIT,
        token=True,
    )

    print(f"Original number of questions: {len(dataset)}")

    sample_size = min(NUM_SAMPLES, len(dataset))

    # Reproducible random downsampling.
    dataset = (
        dataset
        .shuffle(seed=RANDOM_SEED)
        .select(range(sample_size))
    )

    print(
        f"Selected {len(dataset)} questions "
        f"with random seed {RANDOM_SEED}."
    )

    annotations = []
    failed_rows = []

    for sample_index, row in enumerate(
        tqdm(dataset, desc="Exporting RadImageNet-VQA")
    ):
        try:
            image = row.get("image")

            if image is None:
                raise ValueError("The image field is empty.")

            question = str(row.get("question") or "").strip()

            if not question:
                raise ValueError("The question field is empty.")

            raw_answer = row.get("answer")
            raw_choices = row.get("choices")
            raw_question_type = row.get("question_type")
            raw_metadata = row.get("metadata")

            metadata = normalize_metadata(raw_metadata)
            choices = normalize_choices(raw_choices)
            question_type = normalize_question_type(
                raw_question_type
            )

            formatted_prompt = format_question(
                question=question,
                choices=choices,
                question_type=question_type,
            )

            ground_truth = resolve_ground_truth(
                raw_answer=raw_answer,
                choices=choices,
                metadata=metadata,
            )

            if not ground_truth:
                raise ValueError(
                    "Could not determine a non-empty ground-truth answer."
                )

            # No top-level question ID is shown in this benchmark,
            # so create a deterministic ID within the sampled subset.
            question_id = (
                f"radimagenet-vqa-{SPLIT}-{sample_index:06d}"
            )

            # Using the question ID guarantees one unique image path
            # for every exported QA entry.
            image_filename = (
                f"{sanitize_filename(question_id)}"
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
                    "prompt": formatted_prompt,
                    "GT": ground_truth,
                    "category": get_category(metadata),
                }
            )

        except Exception as error:
            failed_rows.append(
                {
                    "sample_index": sample_index,
                    "question": row.get("question"),
                    "question_type": row.get("question_type"),
                    "answer": row.get("answer"),
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
        with FAILED_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                failed_rows,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Failed:      {len(failed_rows)} rows")
        print(f"Failure log: {FAILED_PATH.resolve()}")


if __name__ == "__main__":
    main()