from Model_RNN.utils import evaluate_model,Dataset
from Model_RNN.model import RNN
from Model_RNN.config import Config
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__ == '__main__':
    config = Config()
    train_file = "data/cdiscount_train.csv.zip"

    w2v_file = 'C:\DEV\Article\data\cc.fr.300.vec'

    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file)

