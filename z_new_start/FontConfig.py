class Config:
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.model_save_path = './model.pth'
        self.vocab_size = 10000
        self.embed_size = 256
        self.num_heads = 8
        self.num_layers = 6
        self.dropout = 0.1
