import csv
import glob

import mxnet as mx
from bert_embedding import BertEmbedding
from textFeature.textFeatures import DecoderRNN as decoder, EncoderCNN as encoder
from gcn.gcn import Model as stgcn
import pickle
import argparse
import torch
import numpy as np
from numpy.lib.format import open_memmap

from pytorch_pretrained_bert import BertTokenizer, BertModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

dataRootPath='/home/lee/Downloads/MELD.Raw/train/'
csvPath=dataRootPath+'train_sent_emo.csv'
vidPath=dataRootPath+'train_splits/*.mp4'
vidList=glob.glob(vidPath)

dataGenerationFLAG=False # True , False
batchSize=64
num_epoch=200
num_iteration=0
device=torch.device('cuda')
max_utt_length=0
if dataGenerationFLAG:
    ''''''
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
    # Predict hidden states features for each layer
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    model.to('cuda')

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

        embeddingsList = []
        tokenCountList = []
        total_token_count = 0
        total_token_count = 103175

        num_iteration = len(utts)  # of utt = 9989, # of speakers = 260, # of node = 15140
        # print('num_iteration:', num_iteration) # 157
        '''
        fp = open_memmap(
            './train.npy',
            dtype='float32',
            mode='w+',
            shape=87061248)
        '''
        fp2 = open_memmap(
            './tokenSizeList.npy',
            dtype='int32',
            mode='w+',
            shape=(1, num_iteration))
        for j in range(num_iteration):
            text = utts[j:j + 1]
            # print(j + 1, 'input.shape:', len(text)) #text.shape)
            # input = torch.tensor(input).to(device)
            '''
            # zero the gradients
            decoder.zero_grad()
            encoder.zero_grad()

            # set decoder and encoder into train mode
            encoder.train()  # CNN == conv 64 -conv 64 -conv 64?-maxpool, embedding size is 64?? maybe batch size!
            decoder.train()  # biLSTM-biLSTM

            # Pass the inputs through the CNN-RNN model.
            features = encoder(input) #CNN
            outputs = decoder(features, None) #captions_train  #LSTM
            #gcn
            #stgcn_model=stgcn(in_channels=2, num_class=7,graph_args=None, edge_importance_weighting=None)
            # graph ? -> Aij= 1- arccos(sim(ui, uj))/3.14  or c/Freq, speaking coefficient c == 5
            # node = 15140(== N of Vs )
            # lr = 0.01
            # dropout=0.5
            # L2 loss weight = 5e-4
            # epoch = 200
            # optim = Adam
            '''
            # TODO : 수/목/ modify gcn/encoder/decoder architecture!
            # TODO : 금~월 : training and debuging!

            # Load pre-trained model tokenizer (vocabulary)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Tokenized input
            # text = "[CLS] Um ok uh oh god um when you and uh Ross first started going out it was really hard for me um
            # for many reasons which I'm not gonna bore you with now but um I just I see how happy he is you know and how
            # good you guys are together and um Monica's always saying how nice you are and god I hate it when she's right [SEP]"
            # "[CLS] Um, ok, uh, oh god, um, when you and uh Ross first started going out, it was really hard for me, um,
            # for many reasons, which I'm not gonna bore you with now,[SEP] but um, I just, I see how happy he is, you know,
            # and how good you guys are together, and um, Monica's always saying how nice you are, and god I hate it when she's right [SEP]"
            # "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
            tokenized_text = tokenizer.tokenize(text[0])

            # Mask a token that we will try to predict back with `BertForMaskedLM`
            # masked_index = 8
            # tokenized_text[masked_index] = '[MASK]'
            # assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            # segments_ids = #[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
            segments_ids = [0] * len(indexed_tokens)
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensors = segments_tensors.to('cuda')

            with torch.no_grad():
                # a, b = model(tokens_tensor, segments_tensors)
                embeddings, encoded_layers, c = model(tokens_tensor, segments_tensors)
                embeddingsList.append(embeddings.detach().cpu().numpy()[0])
                tokenCountList.append(embeddings.shape[1])
                total_token_count += embeddings.shape[1]
            print("token size:", embeddings.shape[1])

        print("total_token_count :", total_token_count)

        totalMemSize = 0

        for k, e in enumerate(embeddingsList):
            totalMemSize += np.ndarray.flatten(e).shape[0]
        fp3 = open_memmap(
            './train.npy',
            dtype='float32',
            mode='w+',
            shape=(1, 87061248))
        offset = 0
        for k, e in enumerate(embeddingsList):
            fp3[0, offset:offset + (tokenCountList[k] * 768)] = np.ndarray.flatten(e).shape[0]
            offset += tokenCountList[k] * 768
            fp2[0,k] = tokenCountList[k]

        #utts, speakerIDs, emotionIDs, sentimentIDs, int(dialogueIDs), int(utteranceIDs)
    '''
    ctx = mx.gpu(0)
    bert_embedding = BertEmbedding(ctx=ctx)
    result = bert_embedding(utts)

    for r in result:
        print(len(r[1]))
        #if max_utt_length < len(r[1]):
        #    max_utt_length = len(r[1])
        max_utt_length+=len(r[1])
    print('max_utt_length : ', max_utt_length)
    fp = open_memmap(
        './train.npy',
        dtype='float32',
        mode='w+',
        shape=(len(result), max_utt_length, 768))

    for i, r in enumerate(result):
        word_count_in_utterance = len(result[i][1])
        fp[i, 0:word_count_in_utterance, :] = r[1]
    '''
#with open(label_out_path, 'wb') as f:
#    pickle.dump((sample_name, list(sample_label)), f)

traindata = np.load('./train.npy')
tokenSizeList = np.load('./tokenSizeList.npy')


# len(result) #9989
# result[0][0] == N tuples or lists <- actual tokened list of subwords
# result[0][0][768] == Nx768 <- actual tokened  of subwords
# word_count_in_utterance=len(result[9989][1])
# result[9989][1][word_count_in_utterance][768]#word_count_in_utterance <= dynamically change
# 32 batch
# TODO : save word embeddings to npy or pickle and reuse it!

################## BertModel

# Load pre-trained model (weights)
#model = BertModel.from_pretrained('bert-base-uncased')
#model.eval()
#model.to('cuda')

utts = []

speakers = []
speakerIDs = []

emotions = []
emotionIDs = []

sentiments = []
sentimentIDs = []

dialogueIDs = []
utteranceIDs = []

count = 0

with open(csvPath, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        count += 1

        if count == 1:
            continue
        utts.append(row[1])
        speakers.append(row[2])
        emotions.append(row[3])
        sentiments.append(row[4])
        dialogueIDs.append(int(row[5]))
        utteranceIDs.append(int(row[6]))

    speakersNetList = list(set(speakers))
    emotionsNetList = list(set(emotions))
    sentimentsNetList = list(set(sentiments))

    for s in speakers:
        speakerIDs.append(speakersNetList.index(s))
    for e in emotions:
        emotionIDs.append(emotionsNetList.index(e))
    for s in sentiments:
        sentimentIDs.append(sentimentsNetList.index(s))

    print('----------------------------------------------')

num_iteration=len(utts)
total_token_count = 0
num_epoch = 1
offset=0
for i in range(num_epoch):
    for j in range(num_iteration):

        tokenSize=tokenSizeList[0,j]
        text = traindata[0,offset:offset+tokenSize*768]
        offset+=tokenSize*768

        print(j + 1, 'input.shape:', len(text)) #text.shape)
        # input = torch.tensor(input).to(device)
        '''
        # zero the gradients
        decoder.zero_grad()
        encoder.zero_grad()

        # set decoder and encoder into train mode
        encoder.train()  # CNN == conv 64 -conv 64 -conv 64?-maxpool, embedding size is 64?? maybe batch size!
        decoder.train()  # biLSTM-biLSTM

        # Pass the inputs through the CNN-RNN model.
        features = encoder(input) #CNN
        outputs = decoder(features, None) #captions_train  #LSTM
        #gcn
        #stgcn_model=stgcn(in_channels=2, num_class=7,graph_args=None, edge_importance_weighting=None)
        # graph ? -> Aij= 1- arccos(sim(ui, uj))/3.14  or c/Freq, speaking coefficient c == 5
        # node = 15140(== N of Vs )
        # lr = 0.01
        # dropout=0.5
        # L2 loss weight = 5e-4
        # epoch = 200
        # optim = Adam
        '''