import sklearn.model_selection as sms
import pandas as pd
import numpy as np
import torch
from torchtext import data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder


def get_embedding_matrix(vocab_chars):
    # one hot embedding plus all-zero vector
    vocabulary_size = len(vocab_chars)
    onehot_matrix = np.eye(vocabulary_size, vocabulary_size)
    return onehot_matrix

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.vocab = None
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
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

    def load_data(self,train_file):
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

        tokenizer = lambda sent: list(sent[::-1])

        # Creating Field for data
        TEXT = data.Field(tokenize=tokenizer, lower=True, fix_length=self.config.seq_len)
        LABEL = data.Field(sequential=False, use_vocab=False)


        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df, val_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)


        val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
        val_data = data.Dataset(val_examples, datafields)

        TEXT.build_vocab(train_data)

        embedding_mat = get_embedding_matrix(list(TEXT.vocab.stoi.keys()))
        TEXT.vocab.set_vectors(TEXT.vocab.stoi, torch.FloatTensor(embedding_mat), len(TEXT.vocab.stoi))
        self.vocab = TEXT.vocab
        self.embeddings = TEXT.vocab.vectors






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