import argparse
import json
import os

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


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
    print("Multi-Med: Qwen2-VL inference")
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


    if model_path.startswith("Qwen/Qwen3-VL-"):
        model = (
            Qwen3VLForConditionalGeneration
            .from_pretrained(
                model_path,
                **model_kwargs,
            )
        )
    elif model_path.startswith("Qwen/Qwen2-VL-"):
        model = (
            Qwen2VLForConditionalGeneration
            .from_pretrained(
                model_path,
                **model_kwargs,
            )
        )
    else:
        raise ValueError(
            f"Unsupported model path: {model_path}"
        )

    model.eval()

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor_kwargs = {}

    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = (
            args.min_pixels
        )

    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = (
            args.max_pixels
        )

    processor = AutoProcessor.from_pretrained(
        model_path,
        **processor_kwargs,
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
            # SAME prompt function used by LLaVA.
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
            # Qwen2-VL chat message
            # =================================================

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": question_prompt,
                        },
                    ],
                }
            ]

            # =================================================
            # Qwen2-VL preprocessing
            #
            # Follow the official Qwen2-VL pipeline:
            #
            #   apply_chat_template(tokenize=False)
            #       ↓
            #   process_vision_info()
            #       ↓
            #   processor(...)
            # =================================================

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = (
                process_vision_info(
                    messages
                )
            )

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # -------------------------------------------------
            # With device_map="auto", model.device points to
            # the embedding/input device for standard Qwen2-VL
            # inference.
            # -------------------------------------------------

            inputs = inputs.to(
                model.device
            )

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

            # -------------------------------------------------
            # Deterministic benchmark evaluation
            # -------------------------------------------------

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

                generated_ids = model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            # =================================================
            # IMPORTANT:
            # Remove input prompt before decoding.
            #
            # Without this, extract_prediction() could parse
            # option letters from the MCQ prompt itself.
            # =================================================

            generated_ids_trimmed = [
                out_ids[
                    len(in_ids):
                ]
                for in_ids, out_ids
                in zip(
                    inputs.input_ids,
                    generated_ids,
                )
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # =================================================
            # Prediction parsing
            #
            # SAME implementation as LLaVA.
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
            # SAME JSON schema as LLaVA
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
            "Qwen-VL inference and evaluation "
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
        default="Qwen/Qwen2-VL-7B-Instruct",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
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
        help=(
            "Optional attention implementation, "
            "e.g. flash_attention_2."
        ),
    )

    # ========================================================
    # Qwen2-VL image resolution
    # ========================================================

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help=(
            "Optional minimum image pixels used "
            "by the Qwen2-VL processor."
        ),
    )

    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help=(
            "Optional maximum image pixels used "
            "by the Qwen2-VL processor."
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
            "outputs/qwen2vl_predictions.jsonl"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default=(
            "outputs/qwen2vl_report"
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

        # SAME evaluator as LLaVA.
        run_evaluation(args)

    elif args.stage == "all":

        if args.num_chunks != 1:

            print(
                "\nWARNING: --stage all with "
                "--num-chunks > 1 will evaluate only "
                "the current chunk.\n"
            )

        run_inference(args)

        # SAME evaluator as LLaVA.
        run_evaluation(args)