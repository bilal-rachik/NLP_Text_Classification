from Model_fastText.utils import evaluate_model,Dataset
from Model_fastText.model import fastText
from Model_fastText.config import Config
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__ == '__main__':
    config = Config()
    train_file = "../data/cdiscount_train.csv.zip"

    w2v_file = 'C:\DEV\Article\data\cc.fr.300.vec'

    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file)

    # Create Model with specified optimizer and loss function
    ##############################################################
    model = fastText(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=config.lr)

    #NLLLoss = nn.NLLLoss()
    CrossEntropyLoss = nn.CrossEntropyLoss()


    model.add_optimizer(optimizer)
    model.add_loss_op(CrossEntropyLoss)

    ##############################################################

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)


    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))


