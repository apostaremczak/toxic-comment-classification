from sklearn.model_selection import train_test_split

from comment_tf_record import create_data_record
from preprocessing import load_training_data, tokenize_comments
from model.classifier import get_tokenizer

TEST_FRACTION = 0.01


def main():
    tokenizer = get_tokenizer()

    # Load and tokenize inputs
    comments, comment_labels = load_training_data()
    comment_ids, comment_masks = tokenize_comments(comments, tokenizer)

    # Split data into train and test tests
    input_train, input_test, mask_train, mask_test, label_train, label_test = train_test_split(
        comment_ids, comment_masks, comment_labels, test_size=TEST_FRACTION,
        shuffle=True)

    # Save tokenized data into TF record files for easier later use
    create_data_record(input_train, mask_train, label_train, "data/train")
    create_data_record(input_test, mask_test, label_test, "data/test")


if __name__ == '__main__':
    main()
