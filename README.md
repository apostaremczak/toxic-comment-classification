# Toxic Comment Classification

Identify and classify toxic online comments.

**Deadline:** 07.06.2020, 23:59

## Requirements

* dataset - any
* score - accuracy
* trained model should be hosted using `tf.serving`
* script for submitting data to a hosted classification service
* instruction on submitting test data

### Data format

Test data will be sent as a CSV file, with each sentence starting on a new line.

## Dataset

Train data comes from 
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
> * identity_hate

>You must create a model which predicts a probability of each type of toxicity for each comment.
