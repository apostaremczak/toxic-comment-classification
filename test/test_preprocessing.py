import numpy as np
import pandas as pd
import unittest

from model.classifier import get_tokenizer
from utils import preprocessing


class TestPreprocessing(unittest.TestCase):
    test_data = pd.read_csv("test_data.csv")

    def test_binarize_toxicity_labels(self):
        desired_labels = np.array([0, 1, 1])
        result_labels = preprocessing._binarize_toxicity_labels(
            TestPreprocessing.test_data)["is_any_toxic"] \
            .to_numpy()
        self.assertTrue(np.all(desired_labels == result_labels))

    def test_tokenize_comments(self):
        comments = TestPreprocessing.test_data["comment_text"]
        tokenizer = get_tokenizer(max_seq_length=10)
        word_ids = [
            [12476, 999, 8699, 7615, 999],
            [22017, 999],
            [2061, 11704]
        ]

        expected_word_ids = np.array([
            [101] + ids + [102] + [0] * (8 - len(ids)) for ids in word_ids
        ], dtype="int32")

        tokenized_ids, _ = preprocessing.tokenize_comments(comments,
                                                           tokenizer,
                                                           max_seq_len=10)
        self.assertTrue(np.all(expected_word_ids == tokenized_ids))


if __name__ == '__main__':
    unittest.main()
