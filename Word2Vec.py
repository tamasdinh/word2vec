#%%
import os
from collections import Counter
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#%%
with open(os.getcwd() + '/data/text8') as f:
    text = f.read()
print(text[:100])


#%%
def text_preprocess(text, occ_limit=5):
    '''
    Light preprocessing of text for NLP purposes (punctuation encoding, frequency filtering)
    :param text: simple text format variable
    :param occ_limit: limit for word frequency (all words are kept that have a frequency higher than the limit)
    :return: list - cleaned text split by space
    '''
    punct_dict = {
        '.': ' <PERIOD> ',
        ',': ' <COMMA> ',
        '"': ' <QUOTATION_MARK> ',
        ';': ' <SEMICOLON> ',
        '!': ' <EXCLAMATION_MARK> ',
        '?': ' <QUESTION_MARK> ',
        '(': ' <LEFT_PAREN> ',
        ')': ' <RIGHT_PAREN> ',
        '--': ' <HYPHENS> ',
        '\\n': ' <NEWLINE> ',
        ':': ' <COLON> '
    }
    text = text.lower()
    for item in punct_dict.items():
        text = text.replace(item[0], item[1])
    text = text.split()
    word_counts = Counter(text)
    trimmed_words = [word for word in text if word_counts[word] > occ_limit]

    return trimmed_words


#%%
words = text_preprocess(text)
print(words[:100])

#%%
# Take some stats on the text
print('Total number of words in text: {}'.format(len(words)))
print('Number of unique words in text: {}'.format(len(set(words))))


#%%
def create_lookup_tables(word_list):
    """
    Creates 2 dictionaries for vocabulary encoding: int2word and word2int
    :param word_list: list of words
    :return: 2 dictionaries: int2word and word2int
    """
    word_counts = Counter(word_list)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int2word = {index: word for index, word in enumerate(sorted_vocab)}
    word2int = {word: index for index, word in int2word.items()}

    return int2word, word2int


#%%
int2word, word2int = create_lookup_tables(words)
int_words = [word2int[word] for word in words]
print(int_words[:30])

#%%
threshold = 1e-5
word_counts = Counter(int_words)
total_cnts = len(int_words)

freqs = {word: count / total_cnts for word, count in word_counts.items()}
del_probs = {word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}

train_words = [word for word in int_words if np.random.random() < 1 - del_probs[word]]
print(train_words[:100])


#%%
def get_context(words, idx, window_size=5):
    """
    Implements windowing according to the skip-gram methodology defined by Mikolov et al. - given the window size C,
    the function generates a random number R in [1:C] and then takes the preceding R and the subsequent R words for
    word with 'idx' index in the list 'words' provided as input to the function.
    :param words: list of words from training text
    :param idx: index of word of interest in param words
    :param window_size: window size C
    :return: list of contexts words for idx in words
    """
    r = np.random.randint(1, window_size+1)
    first_idx = idx - r if (idx-r) > 0 else 0
    last_idx = idx + r + 1
    target_words = words[first_idx:idx] + words[idx+1:last_idx]

    return target_words


#%%
def get_batches(words, batch_size, window_size=5):
    """
    Generates batches of 'batch_size' one-by-one from list 'words' provided as input
    :param words: list of words in text to be analyzed, cleaned
    :param batch_size: size of each batch
    :param window_size: parameter for skip-gram word context identification
    :return: input words and corresponding ground-truth outputs for model
    """
    n_batches = len(words) // batch_size

    words = words[:n_batches*batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_context(batch, ii, window_size=5)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))

        yield x, y


#%%
# Testing the batching
int_text = [i for i in range(20)]
x, y = next(get_batches(int_text, batch_size=4))

print(int_text)
print('x\n', x)
print('y\n', y)


#%%
class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()

        self.embed = nn.Embedding(n_vocab, n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        log_ps = self.log_softmax(scores)

        return log_ps


#%%
def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """
    Calculates the cosine similarity (the cosine of the angle) between 2 embedding vectors.
    Samples validation words (both frequent and infrequent) for similarity calculation.
    :param embedding: embedding layer from trained embedding neural network
    :param valid_size: length of validation word vector
    :param valid_window: size of radius around word of interest for embedding check
    :param device: 'cpu' or 'cuda' to be used in computation
    :return: sampled validation examples, similarity scores
    """
    embed_vectors = embedding.weight

    # norm of embedding vectors
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # sampling n words from ranges (0, window) and (1000, 1000 + window). Lower IDs are more frequent
    valid_examples = np.array(np.random.sample(range(valid_window)), valid_size // 2)
    valid_examples = np.append(valid_examples, np.random.sample(range(1000, 1000 + valid_window), valid_size // 2))
    valid_examples = torch.LongTensor(valid_examples).to(device)

    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t()) / magnitudes

    return valid_examples, similarities


#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Training on {}'.format('CUDA' if device == 'cuda' else 'CPU'))

embedding_dim = 300
learning_rate = 0.003

model = SkipGram(len(word2int), embedding_dim).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

print_every = 500
steps = 0
epochs = 1

#%%
for e in range(epochs):

    for inputs, targets in get_batches(train_words, 512):
        steps += 1
        inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
        inputs, targets = inputs.to(device), targets.to(device)

        log_ps = model(inputs)
        loss = criterion(log_ps, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % print_every == 0:
            valid_examples, similarities = cosine_similarity(model.embed, device=device)
            _, closest_idxs = valid_similarities.topk(6)

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int2word[idx.item()] for idx in closest_idxs[ii][1:]]
                print(int2word[valid_idx.item()] + '|' + ', '.join(closest_words))
            print(...)

#%%
embeddings = model.embed.weight.to('cpu').data.numpy()
words_to_chart = 600
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:words_to_chart, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(words_to_chart):
    plt.scatter(*embed_tsne[idx, :], color='blue')
    plt.annotate(int2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
