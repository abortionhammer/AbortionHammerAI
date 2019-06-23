# AbortionHammerAI
AI program that recognizes pro-life content on Twitter and tag the tweets with specific messages.

## Installation
In the AbortionHammerAI folder, execute the following commands:
```
python -m venv venv
venv\Scripts\activate
cd src
pip install -r requirements.txt
pip install -r pytorch-requirements.txt

Torch for Windows:
pip install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp36-cp36m-win_amd64.whl (or other version)

python -m spacy download en
```

## Tweets Download
The file **src/download-tweets.py** contains several functions to download/stream tweets.  
Before using it, please create the configuration file **config/userinfo.txt** structured this way:
```
Twitter consumer key
Twitter consumer secret
Twitter access token
Twitter access token_secret
Twitter oauth callback url (urlencoded)
```

## Datasets
The datasets used by the program are:  
Dataset | File Path
--- | ---
Training | datasets/training/training-data.txt
Testing | datasets/testing/testing-data.txt


## Train Model
Use this command to train the model:
```
python train_stance.py --train_model
```
It will create a file **save/best_params** that contains the trained weights.

### Additional Parameters:
... under construction ...

## Predict
Predict the stances of the testing dataset:
```
python train_stance.py --submit
```
The predicted stances of the testing dataset will be written in the file **results/predicted.txt**.

... under construction ...

## References
Transfer Learning in NLP for Tweet Stance Classification, by Prashanth Rao, 15 January 2019  
https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde  
Related GitHub repository:  
https://github.com/prrao87/tweet-stance-prediction  
