import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import json
from torchvision import transforms, utils


# char_dict needs to be defined wherever this is used
def oneHotEncode(char, char_dict):
    ret = torch.zeros([1, len(char_dict)])
    ret[0][char_dict[char]] = 1
    return ret


def encode_chunk(chunk, char_dict):
    chunk_len = len(chunk)
    ret = torch.zeros(chunk_len,1,len(char_dict))
    
    for i,char in enumerate(chunk):
        ret[i] = oneHotEncode(char, char_dict) 
    return ret

 
def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


class loader(Dataset):
    # TODO: modify this to read JSON files from genius lyrics data
    def __init__(self, file_path, with_tags=False):
        
        with open(file_path) as json_data:
            data = json.load(json_data)
        
        content = ''
        self.songs = []
        
        for song in data['songs']:
            if with_tags:
                content += '<start>\n' + 'Title: ' + song['title'] + '\n' + song['lyrics'] + '\n<end>\n'
                self.songs.append('<start>\n' + 'Title: ' + song['title'] + '\n' + song['lyrics'] + '\n<end>\n')
            else:
                content += 'Title: ' +  song['title'] + song['lyrics']
                self.songs.append('Title: ' +  song['title'] + '\n' +song['lyrics'])
                
        unique_chars = sorted(list(set(content)))
        self.char_dict = dict(zip(unique_chars, range(len(unique_chars))))
        
    def __len__(self):
        # get number of songs in dataset
        return len(self.songs)

    def __getitem__(self, idx):
        # retrieve a song at idx
        return self.songs[idx]

    
class Nnet(nn.Module):
    def __init__(self, output_size, hidden_dim, n_layers, dropout=0):
        super(Nnet, self).__init__()
        self.input_dim = output_size
        self.output_dim = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # architecture
        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, dropout=dropout)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
              

    def forward(self, x, hidden, heatmap=False):
        batch_size = x.size(0)
        lstm_out, hidden = self.lstm_layer(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out)
        if heatmap:
            return lstm_out, out, hidden
        else:
            return out, hidden
    
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden