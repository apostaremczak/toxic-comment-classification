import tensorflow as tf
from transformers import DistilBertConfig, DistilBertTokenizer, \
    TFDistilBertModel

from model.config import CommentClassifierConfig
from utils.constants import MAX_SEQ_LENGTH, NUM_CLASSES

MODEL_NAME = "distilbert-base-uncased"


def get_tokenizer(max_seq_length: int = MAX_SEQ_LENGTH) -> DistilBertTokenizer:
    return DistilBertTokenizer.from_pretrained(
        MODEL_NAME,
        do_lower_case=True,
        max_length=max_seq_length,
        pad_to_max_length=True,
        add_special_tokens=True
    )


def create_model(model_config: CommentClassifierConfig,
                 saved_weights_path: str = None,
                 max_seq_length: int = MAX_SEQ_LENGTH) -> tf.keras.Model:
    """
    :param model_config:       CommentClassifierConfig
    :param saved_weights_path: If defined, model weights will be loaded
                               from the provided checkpoint path
    :param max_seq_length:     Maximum length of the tokenized input to BERT
    :return:
        Model for text classification using DistilBert transformers
    """
    # Load pre-trained DistilBERT
    bert_config = DistilBertConfig(
        dropout=model_config.bert_dropout,
        attention_dropout=model_config.bert_attention_dropout,
        num_labels=NUM_CLASSES)
    bert_config.output_hidden_states = False
    transformer_model = TFDistilBertModel.from_pretrained(MODEL_NAME,
                                                          config=bert_config)

    input_ids_in = tf.keras.layers.Input(shape=(max_seq_length,),
                                         name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_seq_length,),
                                           name='masked_token', dtype='int32')

    embedding_layer = transformer_model(input_ids_in,
                                        attention_mask=input_masks_in)[0]

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            model_config.lstm_units,
            return_sequences=True,
            dropout=model_config.lstm_dropout,
            recurrent_dropout=model_config.lstm_recurrent_dropout)
    )(embedding_layer)

    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.Dense(
        model_config.hidden_layer_dim,
        activation=model_config.hidden_layer_activation)(x)

    x = tf.keras.layers.Dropout(model_config.final_layer_dropout)(x)
    x = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation=model_config.final_layer_activation)(x)

    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=x)

    # Use transfer learning only - do not train BERT again
    for layer in model.layers[:3]:
        layer.trainable = False

    # Load weights from a checkpoint, but allow partial matching
    # (e.g. due to a change in the optimizer)
    if saved_weights_path is not None:
        model.load_weights(saved_weights_path).expect_partial()

    return model
