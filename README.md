# Domino Reference Project: Named Entity Recognition

*Disclaimer - Domino Reference Projects are starter kits built by Domino researchers showing how to do various analytics in Domino. They are not part of Domino's product, and are not supported by Domino. Once loaded into your Domino environemnt, they are yours to manage. Feel free to improve them as you see fit. We hope they will be a beneficial tool in your analytical journey!

Named Entity Recognition (NER) is an NLP problem, which involves locating and classifying named entities (people, places, organizations etc.) mentioned in unstructured text. This problem is used in many NLP applications that deal with use-cases like machine translation, information retrieval, chatbots and others.  In this project we fit a BiLSTM-CRF model can using a freely available annotated corpus and Keras.

The dataset used in this project is the [Annotated Corpus for Named Entity Recognition](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/). This dataset is based on the GMB (Groningen Meaning Bank) corpus, and has been tagged, annotated and built specifically to train a classifier to predict named entities such as name, location, etc. 

The assets included in the project are:

* **ner.ipynb** - a notebook that performs exploratory data analysis, data wrangling, hyperparameter optimisation, model training and evaluation. The notebook introduces the usecases and discusses the key techniques needed for implementing an NER classification model.

* **model_train.py** - a training script that can be operationalised and retrain the model on demand or on schedule. The script can be used as a template. The key elements that need to be customized for other datasets are:

    * *load_data* - data ingestion function
    * *pre_process* - data wrangling

The majority of the remaining important parameters are controlled via command line arguments to the script.
    
* **model_api.py** - a scoring function that exposes the persisted model as Model API. The *score* function accepts a string of plain text and outputs the tokenized version of the text with the corresponding IOB tags.

## Dockerfile

This project uses a compute environment based on dominodatalab/base:Ubuntu18_DAD_Py3.7_R3.6_20200508. The only additional libraries needed are *plot_keras_history* and *keras-contrib*. You can add them to a custom compute environment by appending the following lines to the Ubuntu18_DAD_Py3.7_R3.6_20200508 Dockerfile:

Add the following entries to the Dockerfile

```
RUN echo "ubuntu    ALL=NOPASSWD: ALL" >> /etc/sudoers
RUN pip install --upgrade pip
RUN pip install install plot_keras_history && pip install git+https://www.github.com/keras-team/keras-contrib.git
```

## Model API

You can test the Model API using the following observation:

{
  "data": {
    "sentence": "President Obama became the first sitting American president to visit Hiroshima"
  }
}
