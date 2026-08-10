# gpt4o_runner.py

import argparse
import base64
import json
import mimetypes
import os
import time

import shortuuid
from openai import OpenAI
from tqdm import tqdm

from LLaVA_runner import (
    load_questions,
    get_chunk,
    normalize_modality,
    normalize_question_type,
    build_mcq_prompt,
    extract_prediction,
    run_evaluation,
)


# ============================================================
# Utilities
# ============================================================

def get_model_id(model_name):
    return model_name


def encode_image_as_data_url(image_path):
    """
    Convert local image into a base64 data URL accepted
    by the OpenAI Responses API.
    """

    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return (
        f"data:{mime_type};base64,"
        f"{encoded}"
    )


# ============================================================
# API call
# ============================================================

def call_gpt4o(
    client,
    model,
    prompt,
    image_data_url,
    max_output_tokens,
    image_detail,
    max_retries,
):

    last_exception = None

    for attempt in range(
        max_retries
    ):

        try:

            response = client.responses.create(
                model=model,

                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type":
                                    "input_image",

                                "image_url":
                                    image_data_url,

                                "detail":
                                    image_detail,
                            },
                            {
                                "type":
                                    "input_text",

                                "text":
                                    prompt,
                            },
                        ],
                    }
                ],

                max_output_tokens=(
                    max_output_tokens
                ),
            )

            return response.output_text.strip()

        except Exception as e:

            last_exception = e

            wait_time = min(
                2 ** attempt,
                30,
            )

            print(
                f"\nAPI error: {e}"
            )

            print(
                f"Retrying in "
                f"{wait_time}s..."
            )

            time.sleep(
                wait_time
            )

    raise RuntimeError(
        "GPT-4o API call failed after "
        f"{max_retries} attempts."
    ) from last_exception


# ============================================================
# Stage 1: inference
# ============================================================

def run_inference(args):

    print("\n" + "=" * 70)
    print("Multi-Med: GPT-4o inference")
    print("=" * 70 + "\n")

    # --------------------------------------------------------
    # OpenAI client
    # --------------------------------------------------------

    client = OpenAI()

    model_id = get_model_id(
        args.model
    )

    print(
        f"Model: {model_id}"
    )

    # --------------------------------------------------------
    # Benchmark
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
        f"Current chunk: "
        f"{args.chunk_idx}/"
        f"{args.num_chunks}"
    )

    print(
        f"Questions in this chunk: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # Output
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

    image_folder = os.path.expanduser(
        args.image_folder
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with open(
        predictions_file,
        "w",
        encoding="utf-8",
    ) as ans_file:

        for item in tqdm(
            questions,
            total=len(questions),
        ):

            # =================================================
            # SAME benchmark prompt
            # =================================================

            question_prompt = build_mcq_prompt(
                item
            )

            # =================================================
            # Local image -> base64 data URL
            # =================================================

            image_file = item[
                "image_path"
            ]

            full_image_path = os.path.join(
                image_folder,
                image_file,
            )

            image_data_url = (
                encode_image_as_data_url(
                    full_image_path
                )
            )

            # =================================================
            # GPT-4o
            # =================================================

            output_text = call_gpt4o(
                client=client,
                model=args.model,
                prompt=question_prompt,
                image_data_url=(
                    image_data_url
                ),
                max_output_tokens=(
                    args.max_output_tokens
                ),
                image_detail=(
                    args.image_detail
                ),
                max_retries=(
                    args.max_retries
                ),
            )

            # =================================================
            # SAME prediction parser
            # =================================================

            prediction_letter = (
                extract_prediction(
                    output_text,
                    item["options"],
                )
            )

            gold_letter = str(
                item["correct_answer"]
            ).strip().upper()

            correct = (
                prediction_letter
                == gold_letter
            )

            # =================================================
            # SAME result schema
            # =================================================

            result = {
                "question_id":
                    item["question_id"],

                "image_path":
                    item["image_path"],

                "question":
                    item["question"],

                "options":
                    item["options"],

                "correct_answer":
                    gold_letter,

                "prediction":
                    output_text,

                "prediction_letter":
                    prediction_letter,

                "correct":
                    correct,

                "question_type":
                    normalize_question_type(
                        item[
                            "question_type"
                        ]
                    ),

                "modality":
                    normalize_modality(
                        item["modality"]
                    ),

                "answer_id":
                    shortuuid.uuid(),

                "model_id":
                    model_id,
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
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "GPT-4o inference and evaluation "
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
    )

    # ========================================================
    # Model
    # ========================================================

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
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
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--predictions-file",
        type=str,
        default=(
            "outputs/gpt4o_predictions.jsonl"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default=(
            "outputs/gpt4o_report"
        ),
    )

    # ========================================================
    # Chunking
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

    # ========================================================
    # API / generation
    # ========================================================

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--image-detail",
        type=str,
        choices=[
            "auto",
            "low",
            "high",
        ],
        default="auto",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
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