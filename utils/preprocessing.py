import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from typing import Tuple
from transformers import PreTrainedTokenizer

from constants import DATA_DIR, MAX_SEQ_LENGTH

TOXIC_LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult",
                "identity_hate")


def _binarize_toxicity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for binary classification: neutral vs binary.
    If a comment has a non-zero label for any of the `TOXIC_LABELS`,
    it will be classified as toxic.
    """

    def assign_toxicity_label(row):
        return int(sum([row[label] for label in TOXIC_LABELS]) > 0)

    df["is_any_toxic"] = df.apply(assign_toxicity_label, axis=1)
    return df[["comment_text", "is_any_toxic"]]


def _balance_classes(binarized_df: pd.DataFrame,
                     minority_to_majority_ratio: float = 0.6,
                     random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset with binary classes using oversampling
    """
    comments = binarized_df[["comment_text"]].values
    labels = binarized_df["is_any_toxic"]

    sampler = RandomOverSampler(random_state=random_state,
                                sampling_strategy=minority_to_majority_ratio)
    comments, labels = sampler.fit_resample(comments, labels)

    # Log dataset statistics
    neutral_stats, toxic_stats = sorted(Counter(labels).items())
    neutral_count = neutral_stats[1]
    toxic_count = toxic_stats[1]
    total_count = neutral_count + toxic_count
    toxic_frac = toxic_count / total_count

    print(f"Total dataset size: {total_count:,}")
    print(f"Class distribution:"
          f"Neutral = {1.0 - toxic_frac:.2%}, Toxic = {toxic_frac:.2%}")

    return comments.squeeze(), labels


def load_training_data(data_dir: str = DATA_DIR + "raw",
                       translated_sample_count: int = 30000,
                       random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    # Full training dataset from Kaggle
    train_en = pd.read_csv(f"{data_dir}/train.csv")
    # Attempt for data augmentation: use inputs translated from English
    # to different languages and back
    train_de = pd.read_csv(f"{data_dir}/train_de.csv").sample(
        translated_sample_count, random_state=random_state)
    train_es = pd.read_csv(f"{data_dir}/train_es.csv").sample(
        translated_sample_count, random_state=random_state)
    train_fr = pd.read_csv(f"{data_dir}/train_fr.csv").sample(
        translated_sample_count, random_state=random_state)

    # Concatenate the datasets and shuffle them in place
    train = pd.concat([train_en, train_de, train_es, train_fr]).sample(
        frac=1, random_state=random_state)

    # Binarize toxicity labels
    binarized_train = _binarize_toxicity_labels(train)

    return _balance_classes(binarized_train)


def tokenize_comments(
        comments: np.ndarray,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = MAX_SEQ_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param comments:
    :param tokenizer:
    :param max_seq_len:
    :return:
    """
    input_ids, input_masks = [], []

    for comment in tqdm(comments, desc="Tokenizing comments"):
        inputs = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=max_seq_len,
            pad_to_max_length=True,
            return_attention_mask=True
        )
        input_ids.append(inputs["input_ids"])
        input_masks.append(inputs["attention_mask"])

    input_ids = np.asarray(input_ids, dtype='int32')
    input_masks = np.asarray(input_masks, dtype='int32')

    return input_ids, input_masks
