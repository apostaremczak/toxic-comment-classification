from dataclasses import dataclass


@dataclass
class CommentClassifierConfig:
    bert_dropout: float = 0.2
    bert_attention_dropout: float = 0.2
    lstm_units: int = 20
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.2
    hidden_layer_dim: int = 50
    hidden_layer_activation: str = "relu"
    final_layer_dropout: float = 0.3
    final_layer_activation: str = "sigmoid"
