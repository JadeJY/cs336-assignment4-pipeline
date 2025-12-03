from __future__ import annotations

import os
from typing import Any
import fasttext
classify_quality_model = None
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(DIR_PATH, '..', 'cs336_data', 'data', 'classifiers', 'quality_classifier.bin')

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from cs336_data.preprocessing import extract_text_from_html_bytes
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    from cs336_data.preprocessing import identify_language
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    from cs336_data.preprocessing import mask_emails
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    from cs336_data.preprocessing import mask_phone_numbers
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    from cs336_data.preprocessing import mask_ips
    return mask_ips(text)

def run_classify_nsfw(text: str) -> tuple[Any, float]:
    from cs336_data.preprocessing import classify_nsfw_speech
    return classify_nsfw_speech(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    from cs336_data.preprocessing import classify_toxic_speech
    return classify_toxic_speech(text)

# uv run pytest -k test_classify_quality


def run_classify_quality(text: str) -> tuple[Any, float]:
    global classify_quality_model
    if classify_quality_model is None:
        classify_quality_model = fasttext.load_model(model_path)
    clean_text = text.replace('\n', ' ')
    labels, scores = classify_quality_model.predict(clean_text)
    label = labels[0].replace('__label__', '')
    final_label = 'wiki' if label == 'hq' else 'cc'
    score = scores[0]
    return final_label, score


def run_gopher_quality_filter(text: str) -> bool:
    from cs336_data.preprocessing import test_gopher
    return test_gopher(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    from cs336_data.deduplication import exact_line_deduplication
    return exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    from cs336_data.deduplication import minhash_deduplication
    return minhash_deduplication(input_files, num_hashes, 
                                 num_bands, ngrams, jaccard_threshold, output_directory)
