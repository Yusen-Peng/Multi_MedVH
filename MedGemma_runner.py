import argparse
import json
import os

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)


# ============================================================
# Reuse benchmark/evaluation logic from LLaVA runner
# ============================================================

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
# Model utilities
# ============================================================

def resolve_dtype(dtype):
    """
    Convert command-line dtype string to torch dtype.
    """

    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    key = str(dtype).strip().lower()

    if key not in mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype}. "
            f"Choose from {list(mapping.keys())}."
        )

    return mapping[key]


def get_model_id(model_path):
    """
    Human-readable model identifier.
    """

    return os.path.basename(
        model_path.rstrip("/")
    )


# ============================================================
# Stage 1: inference
# ============================================================

def run_inference(args):

    print("\n" + "=" * 70)
    print("Multi-Med: MedGemma inference")
    print("=" * 70 + "\n")

    # --------------------------------------------------------
    # Model path
    # --------------------------------------------------------

    model_path = os.path.expanduser(
        args.model_path
    )

    model_id = get_model_id(
        model_path
    )

    print(f"Model: {model_id}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_kwargs = {
        "torch_dtype": resolve_dtype(
            args.dtype
        ),
        "device_map": args.device_map,
    }

    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = (
            args.attn_implementation
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **model_kwargs,
    )

    model.eval()

    # --------------------------------------------------------
    # Load processor
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        model_path
    )

    # --------------------------------------------------------
    # Load benchmark
    #
    # SAME implementation as LLaVA.
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
        f"{args.chunk_idx}/{args.num_chunks}"
    )

    print(
        f"Questions in this chunk: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # Prediction file
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
    # Inference loop
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
            # Benchmark prompt
            #
            # SAME prompt function used by LLaVA/Qwen2-VL.
            # =================================================

            question_prompt = build_mcq_prompt(
                item
            )

            # =================================================
            # Image
            # =================================================

            image_file = item["image_path"]

            full_image_path = os.path.join(
                image_folder,
                image_file,
            )

            image = Image.open(
                full_image_path
            ).convert("RGB")

            # =================================================
            # MedGemma chat format
            #
            # Keep system prompt fixed across benchmark.
            # =================================================

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": args.system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question_prompt,
                        },
                        {
                            "type": "image",
                            "image": image,
                        },
                    ],
                },
            ]

            # =================================================
            # MedGemma preprocessing
            #
            # Follow the native processor pipeline directly.
            # =================================================

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # -------------------------------------------------
            # Follow the official MedGemma usage:
            #
            #     .to(model.device, dtype=torch.bfloat16)
            #
            # But use the requested dtype rather than hardcode.
            # -------------------------------------------------

            model_dtype = resolve_dtype(
                args.dtype
            )

            if model_dtype == "auto":
                inputs = inputs.to(
                    model.device
                )
            else:
                inputs = inputs.to(
                    model.device,
                    dtype=model_dtype,
                )

            # =================================================
            # Input length
            #
            # Needed so we decode ONLY newly-generated tokens.
            # =================================================

            input_len = inputs[
                "input_ids"
            ].shape[-1]

            # =================================================
            # Generation arguments
            # =================================================

            generation_kwargs = {
                "max_new_tokens":
                    args.max_new_tokens,

                "num_beams":
                    args.num_beams,

                "use_cache":
                    True,
            }

            if args.temperature > 0:

                generation_kwargs[
                    "do_sample"
                ] = True

                generation_kwargs[
                    "temperature"
                ] = args.temperature

                if args.top_p is not None:

                    generation_kwargs[
                        "top_p"
                    ] = args.top_p

            else:

                generation_kwargs[
                    "do_sample"
                ] = False

            # =================================================
            # Generate
            # =================================================

            with torch.inference_mode():

                generation = model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            # =================================================
            # CRITICAL:
            # Decode only generated tokens.
            #
            # Otherwise the MCQ prompt itself would be included,
            # which could confuse extract_prediction().
            # =================================================

            generation = generation[
                0,
                input_len:
            ]

            output_text = processor.decode(
                generation,
                skip_special_tokens=True,
            ).strip()

            # =================================================
            # Prediction parsing
            #
            # SAME implementation as LLaVA/Qwen2-VL.
            # =================================================

            prediction_letter = extract_prediction(
                output_text,
                item["options"],
            )

            gold_letter = str(
                item["correct_answer"]
            ).strip().upper()

            correct = (
                prediction_letter
                == gold_letter
            )

            # =================================================
            # SAME JSON schema as other runners
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
                        item["question_type"]
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
            "MedGemma inference and evaluation "
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
        default="google/medgemma-4b-it",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=[
            "auto",
            "float16",
            "fp16",
            "bfloat16",
            "bf16",
            "float32",
            "fp32",
        ],
    )

    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are an expert medical imaging specialist."
        ),
        help=(
            "System prompt used for every benchmark question."
        ),
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
        help=(
            "Multi-Med JSON or JSONL benchmark file."
        ),
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--predictions-file",
        type=str,
        default=(
            "outputs/medgemma_predictions.jsonl"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default=(
            "outputs/medgemma_report"
        ),
    )

    # ========================================================
    # Chunked inference
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

        # SAME evaluator as LLaVA / Qwen2-VL.
        run_evaluation(args)

    elif args.stage == "all":

        if args.num_chunks != 1:

            print(
                "\nWARNING: --stage all with "
                "--num-chunks > 1 will evaluate only "
                "the current chunk.\n"
            )

        run_inference(args)

        # SAME evaluator as LLaVA / Qwen2-VL.
        run_evaluation(args)