from __future__ import division
from __future__ import print_function

import os
from data_loader import DataLoader
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import random


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/test.png'
    fnCorpus = '../data/corpus.txt'

    fnCharList = '../model/charList2.txt'
    fnAccuracy = '../model/accuracy2.txt'
    fnTrain = '../data2/'
    fnInfer = '../data2/test.png'
    fnCorpus = '../data2/corpus.txt'


def main():
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    SEED = 43
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    "main function"

    batchSize = 1
    imgSize = (1024, 32, 1)
    maxTextLen = 128
    # load training data, create TF model
    charlist = open(FilePaths.fnCharList).read()
    print(charlist)
    loader = DataLoader(FilePaths.fnTrain, batchSize, imgSize, maxTextLen)
    print("loading model")
    model = keras.models.load_model('../models/model-val-best')

    model.summary()

    validation_dataset = loader.getValidationDataSet()

    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(-1).output
    )
    print(model.get_layer(-1).name)
    prediction_model.summary()

    # A utility function to decode the output of the network
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        pred = tf.dtypes.cast(pred, tf.float32)
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :maxTextLen
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(
                loader.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    #  Let's check results on some validation samples
    for batch in validation_dataset.take(100):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(
                loader.num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label.strip())

        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            print(orig_texts[i].strip())
            print(pred_texts[i].strip())

if __name__ == '__main__':
    main()
