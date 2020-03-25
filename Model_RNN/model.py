import torch
from torch import nn
import numpy as np
from Model_RNN.utils import evaluate_model


class RNN(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(RNN, self).__init__()

        """
                Arguments
                ---------
                output_size : 
                hidden_sie : Size of the hidden_state of the LSTM
                vocab_size : Size of the vocabulary containing unique words
                embedding_length : Pre-trained faster text fr  word_embeddings which we will use to create our word_embedding look-up table 

                """


        self.vocab_size = vocab_size
        self.word_embeddings = word_embeddings
        self.config = config

        # Embedding Layer
        self.embeddings = nn.Embedding(self.vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(self.word_embeddings, requires_grad=False)

        self.rnn = nn.RNN(input_size=self.config.embed_size,
                          hidden_size=self.config.hidden_size,
                          num_layers=self.config.hidden_layers,
                          dropout=self.config.dropout_keep,
                          bidirectional=self.config.bidirectional)

        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size * self.config.hidden_layers * (1 + self.config.bidirectional),
            self.config.output_size
        )

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):

        """
                Parameters
                ----------
                input_sentence: input_sentence of shape = (batch_size, num_sequences)

                Returns
                -------
                Output of the softmax layer
        """


        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)

        rnn_out, h_n = self.rnn(embedded_sent)
        final_feature_map = self.dropout(h_n)  # shape=(num_layers * num_directions, 64, hidden_size)

        # Convert input to (64, hidden_size * hidden_layers * num_directions) for linear layer
        final_feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

        return train_losses, val_accuracies