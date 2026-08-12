import argparse
import json
import math
import os
import re
from collections import defaultdict

import pandas as pd
import shortuuid
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from models.llava.conversation import conv_templates
from models.llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from models.llava.model.builder import load_pretrained_model
from models.llava.utils import disable_torch_init


# ============================================================
# Benchmark configuration
# ============================================================

QUESTION_TYPE_ORDER = [
    "baseline",
    "modality_mismatch",
    "incorrect_premise",
    "false_suggestions",
]

MODALITY_ORDER = [
    "CT",
    "MRI",
    "CXR",
    "ECG",
    "Pathology",
]


# ============================================================
# General utilities
# ============================================================

def load_questions(path):
    """
    Supports both:
        benchmark.json
        benchmark.jsonl
    """

    path = os.path.expanduser(path)

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [
                json.loads(line)
                for line in f
                if line.strip()
            ]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected top-level JSON list, got {type(data)}"
        )

    return data


def split_list(lst, n):
    """
    Split list into n approximately equal chunks.
    """
    chunk_size = math.ceil(len(lst) / n)
    return [
        lst[i:i + chunk_size]
        for i in range(0, len(lst), chunk_size)
    ]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)

    if k >= len(chunks):
        raise ValueError(
            f"chunk_idx={k} but only {len(chunks)} chunks exist."
        )

    return chunks[k]


def normalize_modality(modality):
    """
    Normalize modality names for reporting.
    """

    mapping = {
        "ct": "CT",
        "mri": "MRI",
        "cxr": "CXR",
        "ecg": "ECG",
        "pathology": "Pathology",
    }

    value = str(modality).strip()

    return mapping.get(value.lower(), value)


def normalize_question_type(question_type):
    """
    Normalize question-type naming.
    """

    value = str(question_type).strip().lower()

    value = value.replace(" ", "_")
    value = value.replace("-", "_")

    aliases = {
        "false_suggestion": "false_suggestions",
        "false_suggestions": "false_suggestions",
        "incorrect_premise": "incorrect_premise",
        "baseline": "baseline",
        "modality_mismatch": "modality_mismatch"
    }

    return aliases.get(value, value)


# ============================================================
# Prompt construction
# ============================================================

def build_mcq_prompt(item):
    """
    Convert benchmark entry into an MCQ prompt.

    Example:

        What abnormalities does Right middle lung suffer from?

        A. pneumothorax
        B. pleural effusion
        C. mass
        D. no abnormalities

        Answer with only the option letter.
    """

    question = item["question"]
    options = item["options"]

    option_lines = []

    for letter, text in options.items():
        option_lines.append(f"{letter}. {text}")

    options_text = "\n".join(option_lines)

    prompt = (
        f"{question}\n\n"
        f"{options_text}\n\n"
        "Answer the question by selecting the correct option. "
        "Please note that there might be questions with (1) mismatched modalities, (2) incorrect premises, (3) false suggestions."
        "In cases like these, please select the proper option that reflects the situation."
        "Respond with only the option letter."
    )

    return prompt


# ============================================================
# Dataset
# ============================================================

class MultiMedDataset(Dataset):

    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        conv_mode,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):

        item = self.questions[index]

        # ----------------------------------------------------
        # Benchmark uses "image_path"
        # ----------------------------------------------------

        image_file = item["image_path"]

        # ----------------------------------------------------
        # Build MCQ prompt
        # ----------------------------------------------------

        qs = build_mcq_prompt(item)

        if self.model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # ----------------------------------------------------
        # Conversation template
        # ----------------------------------------------------

        conv = conv_templates[self.conv_mode].copy()

        conv.append_message(
            conv.roles[0],
            qs
        )

        conv.append_message(
            conv.roles[1],
            None
        )

        prompt = conv.get_prompt()

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        full_image_path = os.path.join(
            self.image_folder,
            image_file,
        )

        image = Image.open(full_image_path).convert("RGB")

        image_tensor = process_images(
            [image],
            self.image_processor,
            self.model_config,
        )[0]

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )

        return (
            input_ids,
            image_tensor,
            image.size,
        )


def collate_fn(batch):

    input_ids, image_tensors, image_sizes = zip(*batch)

    input_ids = torch.stack(
        input_ids,
        dim=0
    )

    image_tensors = torch.stack(
        image_tensors,
        dim=0
    )

    return (
        input_ids,
        image_tensors,
        image_sizes,
    )


def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    conv_mode,
    batch_size=1,
    num_workers=4,
):

    # LLaVA evaluation generally assumes batch size = 1
    assert batch_size == 1

    dataset = MultiMedDataset(
        questions=questions,
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model_config,
        conv_mode=conv_mode,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )


# ============================================================
# Prediction extraction
# ============================================================

def normalize_text(text):
    text = str(text).strip().lower()

    text = re.sub(r"\s+", " ", text)

    text = text.strip(
        " \n\t.,;:!?()[]{}\"'"
    )

    return text


def extract_prediction(output, options):
    """
    Converts free-form LLaVA output into an option letter.

    Handles outputs such as:

        A
        A.
        (A)
        Answer: A
        The answer is A.
        pneumothorax

    Returns:
        prediction_letter
    """

    if output is None:
        return None

    text = str(output).strip()

    valid_letters = [
        str(k).upper()
        for k in options.keys()
    ]

    # --------------------------------------------------------
    # 1. Exact answer such as "A", "A.", "(A)"
    # --------------------------------------------------------

    match = re.match(
        r"^\s*[\(\[]?([A-Za-z])[\)\]\.\:\s]*$",
        text
    )

    if match:
        candidate = match.group(1).upper()

        if candidate in valid_letters:
            return candidate

    # --------------------------------------------------------
    # 2. Common answer patterns
    # --------------------------------------------------------

    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([A-Za-z])[\)\]]?",
        r"(?:the\s+)?correct\s+(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([A-Za-z])[\)\]]?",
        r"^\s*[\(\[]?([A-Za-z])[\)\]\.\:]",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            candidate = match.group(1).upper()

            if candidate in valid_letters:
                return candidate

    # --------------------------------------------------------
    # 3. Standalone letter
    # --------------------------------------------------------

    matches = re.findall(
        r"\b([A-Za-z])\b",
        text,
    )

    for candidate in matches:

        candidate = candidate.upper()

        if candidate in valid_letters:
            return candidate

    # --------------------------------------------------------
    # 4. Model outputs option text instead of letter
    # --------------------------------------------------------

    normalized_output = normalize_text(text)

    # Exact option-text match first
    for letter, option_text in options.items():

        if normalized_output == normalize_text(option_text):
            return str(letter).upper()

    # Then check whether option text appears in generation
    # Prefer longest answer first to avoid substring collisions.
    sorted_options = sorted(
        options.items(),
        key=lambda x: len(str(x[1])),
        reverse=True,
    )

    for letter, option_text in sorted_options:

        normalized_option = normalize_text(option_text)

        if (
            normalized_option
            and normalized_option in normalized_output
        ):
            return str(letter).upper()

    return None


# ============================================================
# Stage 1: inference
# ============================================================

def run_inference(args):

    print("\n" + "=" * 70)
    print("Multi-Med: LLaVA inference")
    print("=" * 70 + "\n")

    disable_torch_init()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_path = os.path.expanduser(
        args.model_path
    )

    model_name = get_model_name_from_path(
        model_path
    )

    print(f"Model: {model_name}")

    tokenizer, model, image_processor, context_len = (
        load_pretrained_model(
            model_path,
            args.model_base,
            model_name,
        )
    )

    # --------------------------------------------------------
    # Original LLaVA convention handling
    # --------------------------------------------------------

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"

        print(
            "Detected plain model. "
            f"Switching conversation mode to {args.conv_mode}."
        )

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    all_questions = load_questions(
        args.question_file
    )

    print(
        f"Total benchmark questions: "
        f"{len(all_questions)}"
    )

    questions = get_chunk(
        all_questions,
        args.num_chunks,
        args.chunk_idx,
    )

    print(
        f"Current chunk: {args.chunk_idx}/"
        f"{args.num_chunks}"
    )

    print(
        f"Questions in this chunk: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    predictions_file = os.path.expanduser(
        args.predictions_file
    )

    output_dir = os.path.dirname(
        predictions_file
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    data_loader = create_data_loader(
        questions=questions,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        conv_mode=args.conv_mode,
        batch_size=1,
        num_workers=args.num_workers,
    )

    # --------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------

    with open(
        predictions_file,
        "w",
        encoding="utf-8",
    ) as ans_file:

        iterator = zip(
            data_loader,
            questions,
        )

        for (
            input_ids,
            image_tensor,
            image_sizes,
        ), item in tqdm(
            iterator,
            total=len(questions),
        ):

            input_ids = input_ids.to(
                device="cuda",
                non_blocking=True,
            )

            # ----------------------------------------------
            # Generate
            # ----------------------------------------------

            with torch.inference_mode():

                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(
                        dtype=torch.float16,
                        device="cuda",
                        non_blocking=True,
                    ),
                    image_sizes=image_sizes,
                    do_sample=(
                        args.temperature > 0
                    ),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            # ----------------------------------------------
            # Decode
            # ----------------------------------------------

            output_text = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )[0].strip()

            prediction_letter = extract_prediction(
                output_text,
                item["options"],
            )

            gold_letter = str(
                item["correct_answer"]
            ).strip().upper()

            correct = (
                prediction_letter == gold_letter
            )

            # ----------------------------------------------
            # IMPORTANT:
            # Preserve all benchmark metadata needed later.
            # ----------------------------------------------

            result = {
                "question_id": item["question_id"],
                "image_path": item["image_path"],

                "question": item["question"],
                "options": item["options"],

                "correct_answer": gold_letter,

                "prediction": output_text,
                "prediction_letter": prediction_letter,

                "correct": correct,

                "question_type": normalize_question_type(
                    item["question_type"]
                ),

                "modality": normalize_modality(
                    item["modality"]
                ),

                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
            }

            ans_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            ans_file.flush()

    print(
        f"\nPredictions saved to:\n"
        f"{predictions_file}\n"
    )

    return predictions_file


# ============================================================
# Prediction loading
# ============================================================

def load_predictions(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# Metric helper
# ============================================================

def accuracy(records):
    """
    Percentage accuracy.
    """

    if len(records) == 0:
        return float("nan")

    correct = sum(
        bool(x["correct"])
        for x in records
    )

    return 100.0 * correct / len(records)


def score_subset(
    records,
    key,
    target,
):

    subset = [
        x
        for x in records
        if x.get(key) == target
    ]

    return accuracy(subset)


# ============================================================
# Stage 2: evaluation/report
# ============================================================

def run_evaluation(args):

    print("\n" + "=" * 70)
    print("Multi-Med: Evaluation")
    print("=" * 70 + "\n")

    predictions_file = os.path.expanduser(
        args.predictions_file
    )

    records = load_predictions(
        predictions_file
    )

    if len(records) == 0:
        raise RuntimeError(
            "Prediction file contains no predictions."
        )

    # --------------------------------------------------------
    # Normalize metadata in case predictions came from
    # different evaluation runs.
    # --------------------------------------------------------

    for x in records:

        x["modality"] = normalize_modality(
            x["modality"]
        )

        x["question_type"] = normalize_question_type(
            x["question_type"]
        )

        # Recompute prediction / correctness when possible.
        # This makes evaluation independent of inference code.
        if (
            "prediction_letter" not in x
            or x["prediction_letter"] is None
        ):

            x["prediction_letter"] = extract_prediction(
                x.get("prediction", ""),
                x["options"],
            )

        x["correct"] = (
            x["prediction_letter"]
            == str(
                x["correct_answer"]
            ).strip().upper()
        )

    # --------------------------------------------------------
    # Detailed statistics
    # --------------------------------------------------------

    total = len(records)

    total_correct = sum(
        x["correct"]
        for x in records
    )

    parse_failures = sum(
        x["prediction_letter"] is None
        for x in records
    )

    print(f"Number of predictions : {total}")
    print(f"Correct               : {total_correct}")

    print(
        f"Overall accuracy       : "
        f"{100 * total_correct / total:.2f}"
    )

    print(
        f"Parsing failures       : "
        f"{parse_failures}"
    )

    print()

    # ========================================================
    # TABLE 3 STYLE RESULT
    # ========================================================

    row = {
        "Model": records[0].get(
            "model_id",
            "LLaVA"
        )
    }

    # --------------------------------------------------------
    # By question type
    # --------------------------------------------------------

    for question_type in QUESTION_TYPE_ORDER:

        row[question_type] = score_subset(
            records,
            "question_type",
            question_type,
        )

    # --------------------------------------------------------
    # By modality
    # --------------------------------------------------------

    for modality in MODALITY_ORDER:

        row[modality] = score_subset(
            records,
            "modality",
            modality,
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    row["Overall"] = accuracy(records)

    result_columns = (
        ["Model"]
        + QUESTION_TYPE_ORDER
        + MODALITY_ORDER
        + ["Overall"]
    )

    results_df = pd.DataFrame(
        [row],
        columns=result_columns,
    )

    # Pretty column names
    display_df = results_df.rename(
        columns={
            "incorrect_premise":
                "incorrect premise",

            "false_suggestions":
                "false suggestion",
        }
    )

    print("=" * 110)
    print("Main Results")
    print("=" * 110)

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
                if pd.notna(x)
                else "-"
            ),
        )
    )

    print()

    # ========================================================
    # Composition / diagnostic tables
    # ========================================================

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Accuracy by question type
    # --------------------------------------------------------

    question_type_rows = []

    for question_type in QUESTION_TYPE_ORDER:

        subset = df[
            df["question_type"]
            == question_type
        ]

        n = len(subset)

        correct = (
            int(subset["correct"].sum())
            if n
            else 0
        )

        acc = (
            correct / n * 100
            if n
            else float("nan")
        )

        question_type_rows.append({
            "question_type":
                question_type,

            "N":
                n,

            "Correct":
                correct,

            "Accuracy":
                acc,
        })

    question_type_df = pd.DataFrame(
        question_type_rows
    )

    print("=" * 70)
    print("Breakdown by Question Type")
    print("=" * 70)

    print(
        question_type_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    print()

    # --------------------------------------------------------
    # Accuracy by modality
    # --------------------------------------------------------

    modality_rows = []

    for modality in MODALITY_ORDER:

        subset = df[
            df["modality"]
            == modality
        ]

        n = len(subset)

        correct = (
            int(subset["correct"].sum())
            if n
            else 0
        )

        acc = (
            correct / n * 100
            if n
            else float("nan")
        )

        modality_rows.append({
            "modality":
                modality,

            "N":
                n,

            "Correct":
                correct,

            "Accuracy":
                acc,
        })

    modality_df = pd.DataFrame(
        modality_rows
    )

    print("=" * 70)
    print("Breakdown by Modality")
    print("=" * 70)

    print(
        modality_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    print()

    # ========================================================
    # FULL modality × question type matrix
    #
    # This is extremely useful for benchmark debugging.
    # ========================================================

    matrix_rows = []

    for modality in MODALITY_ORDER:

        row_matrix = {
            "modality": modality
        }

        for question_type in QUESTION_TYPE_ORDER:

            subset = df[
                (df["modality"] == modality)
                &
                (
                    df["question_type"]
                    == question_type
                )
            ]

            if len(subset):

                score = (
                    subset["correct"].mean()
                    * 100
                )

            else:
                score = float("nan")

            row_matrix[
                question_type
            ] = score

        modality_subset = df[
            df["modality"] == modality
        ]

        row_matrix["Overall"] = (
            modality_subset["correct"].mean()
            * 100
            if len(modality_subset)
            else float("nan")
        )

        matrix_rows.append(
            row_matrix
        )

    matrix_df = pd.DataFrame(
        matrix_rows
    )

    print("=" * 90)
    print("Modality × Question-Type Accuracy Matrix")
    print("=" * 90)

    print(
        matrix_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
                if pd.notna(x)
                else "-"
            ),
        )
    )

    print()

    # ========================================================
    # Save reports
    # ========================================================

    report_dir = os.path.expanduser(
        args.report_dir
    )

    os.makedirs(
        report_dir,
        exist_ok=True,
    )

    # Main paper-style table
    results_df.to_csv(
        os.path.join(
            report_dir,
            "main_results.csv",
        ),
        index=False,
    )

    # Question types
    question_type_df.to_csv(
        os.path.join(
            report_dir,
            "results_by_question_type.csv",
        ),
        index=False,
    )

    # Modalities
    modality_df.to_csv(
        os.path.join(
            report_dir,
            "results_by_modality.csv",
        ),
        index=False,
    )

    # Cross breakdown
    matrix_df.to_csv(
        os.path.join(
            report_dir,
            "results_modality_x_question_type.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Save error cases
    # --------------------------------------------------------

    errors = [
        x
        for x in records
        if not x["correct"]
    ]

    error_file = os.path.join(
        report_dir,
        "errors.jsonl",
    )

    with open(
        error_file,
        "w",
        encoding="utf-8",
    ) as f:

        for x in errors:

            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Parsing failures
    # --------------------------------------------------------

    failures = [
        x
        for x in records
        if x["prediction_letter"] is None
    ]

    failure_file = os.path.join(
        report_dir,
        "parsing_failures.jsonl",
    )

    with open(
        failure_file,
        "w",
        encoding="utf-8",
    ) as f:

        for x in failures:

            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Evaluation reports saved to:\n"
        f"{report_dir}"
    )

    print(
        f"\nErrors: {len(errors)}"
    )

    print(
        f"Parsing failures: {len(failures)}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "LLaVA inference and evaluation "
            "for the Multi-Med benchmark."
        )
    )

    # ========================================================
    # Stage
    # ========================================================

    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "inference",
            "evaluate",
            "all",
        ],
        default="all",
        help=(
            "inference: run model only; "
            "evaluate: evaluate existing predictions; "
            "all: inference + evaluation"
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    parser.add_argument(
        "--model-path",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
    )

    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--conv-mode",
        type=str,
        default="llava_v1",
    )

    # ========================================================
    # Dataset
    # ========================================================

    parser.add_argument(
        "--image-folder",
        type=str,
        default=".",
    )

    parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Multi-Med JSON or JSONL benchmark file.",
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--predictions-file",
        type=str,
        default="outputs/predictions.jsonl",
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default="outputs/report",
    )

    # ========================================================
    # Distributed/chunked inference
    # ========================================================

    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    # ========================================================
    # Generation
    # ========================================================

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "0 is recommended for deterministic "
            "benchmark evaluation."
        ),
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    # ========================================================
    # Execute
    # ========================================================

    if args.stage == "inference":

        run_inference(args)

    elif args.stage == "evaluate":

        run_evaluation(args)

    elif args.stage == "all":

        if args.num_chunks != 1:
            print(
                "\nWARNING: --stage all with "
                "--num-chunks > 1 will evaluate only "
                "the current chunk.\n"
            )

        run_inference(args)
        run_evaluation(args)