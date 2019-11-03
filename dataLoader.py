import csv
import glob

import mxnet as mx
from bert_embedding import BertEmbedding
from .textFeatures import DecoderRNN as decoder, EncoderCNN as encoder
from .gcn import Model as stgcn

import torch
import numpy as np
dataRootPath='/home/lee/Downloads/MELD.Raw/train/'
csvPath=dataRootPath+'train_sent_emo.csv'
vidPath=dataRootPath+'train_splits/*.mp4'
vidList=glob.glob(vidPath)

utts=[]

speakers=[]
speakerIDs=[]

emotions=[]
emotionIDs=[]

sentiments=[]
sentimentIDs=[]

dialogueIDs=[]
utteranceIDs=[]

count=0

with open(csvPath, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        count+=1
        if count==1:
            continue

        utts.append(row[1])
        speakers.append(row[2])
        emotions.append(row[3])
        sentiments.append(row[4])
        dialogueIDs.append(int(row[5]))
        utteranceIDs.append(int(row[6]))

    speakersNetList = list(set(speakers))
    emotionsNetList = list(set(emotions))
    sentimentsNetList=list(set(sentiments))

    for s in speakers:
        speakerIDs.append(speakersNetList.index(s))
    for e in emotions:
        emotionIDs.append(emotionsNetList.index(e))
    for s in sentiments:
        sentimentIDs.append(sentimentsNetList.index(s))
    print('----------------------------------------------')
    #utts, speakerIDs, emotionIDs, sentimentIDs, int(dialogueIDs), int(utteranceIDs)

ctx = mx.gpu(0)
bert_embedding = BertEmbedding(ctx=ctx)
result = bert_embedding(utts)
len(result) #9989
#result[0][0] == N tuples or lists <- actual tokened list of subwords
#result[0][0][768] == Nx768 <- actual tokened  of subwords
# zero the gradients
decoder.zero_grad()
encoder.zero_grad()

# set decoder and encoder into train mode
encoder.train()#CNN == conv 64 -conv 64 -conv 64?-maxpool, embedding size is 64??
decoder.train()#biLSTM-biLSTM
#images = images.to(device)

# Pass the inputs through the CNN-RNN model.
features = encoder(result) #CNN
outputs = decoder(features, None)#captions_train  #LSTM

#gcn
stgcn_model=stgcn(in_channels=2, num_class=7,graph_args=None, edge_importance_weighting=None)
# graph ? -> Aij= 1- arccos(sim(ui, uj))/3.14  or c/Freq, speaking coefficient c == 5
# node = 15140(== N of Vs )

# lr = 0.01
# dropout=0.5
# L2 loss weight = 5e-4
# epoch = 200
# optim = Adam

print("")
