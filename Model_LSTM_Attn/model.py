import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from  Model_LSTM_Attn.utils import evaluate_model


class Lstm_attnClassifier(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(Lstm_attnClassifier, self).__init__()
        self.config = config

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=True)

        # Encoder RNN
        self.lstm = nn.LSTM(input_size=self.config.embed_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.hidden_layers,
                            bidirectional=self.config.bidirectional)

        # Dropout Layer
        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size * (1 + self.config.bidirectional) * self.config.hidden_layers,
            self.config.output_size
        )

        # Softmax non-linearity
        self.softmax = F.log_softmax

    def apply_attention(self, rnn_output, final_hidden_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)  # shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0, 2, 1), soft_attention_weights).squeeze(2)
        return attention_output

    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)

        ##################################### Encoder #######################################
        lstm_output, (h_n, c_n) = self.lstm(embedded_sent)
        # lstm_output.shape = (seq_len, batch_size, num_directions * hidden_size)

        # Final hidden state of last layer (num_directions, batch_size, hidden_size)
        batch_size = h_n.shape[1]
        h_n_final_layer = h_n.view(self.config.hidden_layers,
                                   self.config.bidirectional + 1,
                                   batch_size,
                                   self.config.hidden_size)[-1, :, :, :]

        ##################################### Attention #####################################
        # Convert input to (batch_size, num_directions * hidden_size) for attention
        final_hidden_state = torch.cat([h_n_final_layer[i, :, :] for i in range(h_n_final_layer.shape[0])], dim=1)

        attention_out = self.apply_attention(lstm_output.permute(1, 0, 2), final_hidden_state)
        # Attention_out.shape = (batch_size, num_directions * hidden_size)

        #################################### Linear #########################################
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        final_feature_map = self.dropout(concatenated_vector)  # shape=(batch_size, num_directions * hidden_size)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out,dim=-1)

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
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)) or (epoch == int(3 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label).type(torch.LongTensor)
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