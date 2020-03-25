
class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 32
    bidirectional = True
    output_size = 44
    max_epochs = 5
    lr = 0.25
    batch_size = 64
    max_sen_len = 30 # Sequence length for RNN
    dropout_keep = 0.8