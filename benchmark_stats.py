import json
import pandas as pd

JSON_PATH = "Multi_MedVH_QA/Multi_MedVH_QA.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

modality_mapping = {
    "ct": "CT",
    "mri": "MRI",
    "cxr": "CXR",
    "ecg": "ECG",
    "pathology": "Pathology",
}

df["modality"] = df["modality"].str.lower().map(modality_mapping)
df["question_type"] = df["question_type"].str.lower()

modality_order = [
    "CT",
    "MRI",
    "CXR",
    "ECG",
    "Pathology",
]

question_type_order = [
    "baseline",
    "modality_mismatch",
    "incorrect_premise",
    "false_suggestions",
]

table = pd.crosstab(
    df["modality"],
    df["question_type"],
)

table = table.reindex(
    index=modality_order,
    columns=question_type_order,
    fill_value=0,
)

table["Total"] = table.sum(axis=1)

table.loc["Total"] = table.sum(axis=0)

print("\nBenchmark Composition\n")
print(table)

table.to_csv("results/benchmark_composition.csv")