import csv
import glob

import mxnet as mx
from bert_embedding import BertEmbedding
from mmerc.textFeature.textFeatures import DecoderRNN as decoder, EncoderCNN as encoder
from mmerc.gcn.gcn import Model as stgcn
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap

dataRootPath='/home/lee/Downloads/MELD.Raw/train/'
csvPath=dataRootPath+'train_sent_emo.csv'
vidPath=dataRootPath+'train_splits/*.mp4'
vidList=glob.glob(vidPath)
if 0:
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
    max_word_count=100
    fp = open_memmap(
        './train.npy',
        dtype='float32',
        mode='w+',
        shape=(len(result), 1, max_word_count, 768))

    for i, r in enumerate(result):
        word_count_in_utterance=len(result[i][1])
        fp[i, 0, 0:word_count_in_utterance, :] = r[1]

#with open(label_out_path, 'wb') as f:
#    pickle.dump((sample_name, list(sample_label)), f)
traindata=np.load('./train.npy')
print("")
#len(result) #9989
#result[0][0] == N tuples or lists <- actual tokened list of subwords
#result[0][0][768] == Nx768 <- actual tokened  of subwords
#word_count_in_utterance=len(result[9989][1])
#result[9989][1][word_count_in_utterance][768]#word_count_in_utterance <= dynamically change
#      32batch
#TODO : save word embeddings to npy or pickle and reuse it!
if 0:
    # zero the gradients
    decoder.zero_grad()
    encoder.zero_grad()

    # set decoder and encoder into train mode
    encoder.train()#CNN == conv 64 -conv 64 -conv 64?-maxpool, embedding size is 64?? maybe batch size!
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
