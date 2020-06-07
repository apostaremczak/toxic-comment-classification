# Toxic Comment Classification

Identify and classify toxic online comments.


## Dataset

Training data comes from 
[Kaggle - Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview) 
by Jigsaw/Conversation AI.

### Data description

> You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:
>
>* toxic
>* severe_toxic
>* obscene
>* threat
>* insult
>* identity_hate

>You must create a model which predicts a probability of each type of toxicity for each comment.

## Technical requirements

This project was written in Python 3.8. To install all required dependencies, run

```
pip install -r requirements.txt
```

in your virtual environment.


## Classifying new data

In order run a prediction on the classifier, use `submit_comments.py`.

```
usage: submit_comments.py [-h] input_file output_file

Submit text data to a toxic comment classifier

positional arguments:
  input_file   Path to a CSV file with input data; Each comment should be on a separate line, and the file only one column with no header
  output_file  Path to a CSV file where the model's predictions will be stored

optional arguments:
  -h, --help   show this help message and exit
```

Exemplary use:
```
python3 submit_comments.py example_submission.csv example_output.csv
```

### Data format

Test data must be sent as a CSV file, with each sentence starting on a new line.
