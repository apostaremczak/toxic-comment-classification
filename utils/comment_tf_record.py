import numpy as np
import tensorflow as tf
from tqdm import tqdm

from constants import NUM_CLASSES


def _int64_feature(value: np.ndarray) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value: np.ndarray) -> tf.train.Feature:
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_data_record(inputs: np.ndarray,
                       masks: np.ndarray,
                       labels: np.ndarray,
                       output_file: str):
    # Do one hot encoding on the labels
    one_hot_labels = tf.keras.utils.to_categorical(labels,
                                                   num_classes=NUM_CLASSES)

    output_file = f"{output_file}.tfrecord"
    with tf.io.TFRecordWriter(output_file) as record_writer:
        for (i, comment_id), mask in tqdm(zip(enumerate(inputs), masks),
                                          desc=f"Saving to {output_file}"):
            # Create a Feature from each comment and save it to the file
            feature = {
                "input_token": _int64_feature(comment_id),
                "masked_token": _int64_feature(mask),
                "label": _float_feature(one_hot_labels[i])
            }

            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            record_writer.write(example.SerializeToString())
