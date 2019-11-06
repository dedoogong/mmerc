import torch
from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.word_to_vector import GloVe

encoder = WhitespaceEncoder(["now this ain't funny", "so don't you dare laugh"])

vocab = set(encoder.vocab)
pretrained_embedding = GloVe(name='6B', dim=300, is_include=lambda w: w in vocab)
embedding_weights = torch.Tensor(encoder.vocab_size, pretrained_embedding.dim)
for i, token in enumerate(encoder.vocab):
    embedding_weights[i] = pretrained_embedding[token]
print("")