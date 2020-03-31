import sklearn.model_selection as sms
import pandas as pd
import numpy as np
import torch
from torchtext import data
from torchtext.vocab import Vectors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from spacy.lang.fr import French
from sklearn.feature_extraction.text import strip_accents_ascii
from spacy.lang.fr.stop_words import STOP_WORDS

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
        self.ordinal_encoder = OrdinalEncoder()

    def split_dataset(self,input_path, nb_line, tauxValid):
        data_all = pd.read_csv(input_path, sep=",", nrows=nb_line)
        data_all = data_all[["Description", "Categorie1"]]
        data_all.rename(columns={"Description": "text", "Categorie1": "label"}, inplace=True)
        data_all = data_all.fillna("")
        data_train, data_valid = sms.train_test_split(data_all, test_size=tauxValid, random_state=47)
        return data_train, data_valid

    def get_pandas_df(self, input_path, nb_line=100000, tauxValid=0.2):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''

        data_train, data_valid = self.split_dataset(input_path, nb_line, tauxValid)
        N_train = data_train.shape[0]
        N_valid = data_valid.shape[0]
        print("Train set : %d elements, Validation set : %d elements" % (N_train, N_valid))

        train_labels = self.ordinal_encoder.fit_transform(data_train[["label"]])
        val_labels = self.ordinal_encoder.transform(data_valid[["label"]])
        data_train[["label"]] = train_labels
        data_valid[["label"]] = val_labels

        # converting dtypes using astype
        data_train["label"] = data_train["label"].astype(int)
        data_valid["label"] = data_valid["label"].astype(int)

        data_train["text"] = data_train["text"].astype(str)
        data_valid["text"] = data_valid["text"].astype(str)


        return data_train ,data_valid

    def load_data(self, w2v_file, train_file):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            w2v_file (String): absolute path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''


        NLP = French()

        def tokenizer(sentence):
            # Creating our token object, which is used to create documents with linguistic annotations.
            mytokens = NLP(sentence)

            # Lemmatizing each token and converting each token into lowercase
            mytokens = [word.lemma_.lower() for word in mytokens if word.text != " " and
                        not word.is_punct and not word.like_num and word.text != 'n']
            # Removing stop words
            mytokens = [word for word in mytokens if word not in STOP_WORDS]

            # Remove accentuated char for any unicode symbol
            mytokens = [strip_accents_ascii(word) for word in mytokens]

            # return preprocessed list of tokens
            return mytokens


        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df, val_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)


        val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
        val_data = data.Dataset(val_examples, datafields)

        TEXT.build_vocab(train_data, vectors=Vectors(w2v_file),max_size=20000,min_freq=3)
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab

        self.train_iterator, self.val_iterator = data.BucketIterator.splits(
            (train_data,val_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator):
    model.eval()
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score