import tensorflow as tf

from .constants import MAX_SEQ_LENGTH, NUM_CLASSES

FEATURE_DESCRIPTION = {
    'input_token': tf.io.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
    'masked_token': tf.io.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
    'label': tf.io.FixedLenFeature([NUM_CLASSES], tf.float32),
}


def _parse_comment_record(serialized_example):
    features = tf.io.parse_single_example(serialized_example,
                                          features=FEATURE_DESCRIPTION)

    input_token = features["input_token"]
    mask = features["masked_token"]
    label = features["label"]

    return input_token, mask, label


def load_comment_dataset(tf_record_file_path: str):
    raw_dataset = tf.data.TFRecordDataset(tf_record_file_path)
    return raw_dataset.map(_parse_comment_record)

