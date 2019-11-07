import csv
import glob
import sys

import mxnet as mx
from bert_embedding import BertEmbedding
from textFeature.textFeatures import DecoderRNN as decoder, EncoderCNN as encoder
from gcn.gcn import Model as stgcn # from net.st_gcn import Model as pitcherModel
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


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )

    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

    # processor
    # parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

    # visulize and debug
    # parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
    # parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    # parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--st_weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

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
utts = []
count = 0
with open(csvPath, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        count += 1
        if count == 1:
            continue
        utts.append(row[1])
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
        text=np.reshape(text, (-1, 768))
        print(j + 1, 'input.shape:', len(text)) #text.shape)
        # TODO : 수 - A with edge weighting!! <- similarity 는 seq2vec로 해야겠다!! sim값을 seq2vec로 뽑아서 graph의 edge에 넣은다음 아래와같이 계산하면 됨.
        # TODO : 목 - modify gcn/encoder/decoder architecture!
        # TODO : 금~월 - training and debugging! / biLSTM naver돌리기
        # TODO : 역겨움/무서움 등 안되는건 데이터 불균형 때문이었네. 중립이 너무 압도적 많고, 저런건 너무 너무 거의 없음;
        '''
        if tf_par=="word2vec":
            for u,v,d in dG.edges(data=True):
                if 'w2vec' in d:  
                    # dice = (2*d['weight'])/(dG.node[u]['count']+dG.node[v]['count'])
                    # dG.edge[u][v]['weight'] = dice * (dG.node[u]['count']*dG.node[v]['count'])/((d['w2vec'])**2)

                    # d['weight'] = (dG.node[u]['count']*dG.node[v]['count'])/((1-d['w2vec']))

                    ## angular

                    # dice = (2*d['weight'])/(dG.node[u]['count']+dG.node[v]['count'])
                    # f = (dG.node[u]['count']*dG.node[v]['count'])/(d['w2vec']**2)
                    # print d['w2vec']
                    # d['weight'] = d['weight']/(d['w2vec'])
                    # if u not in counter_word2vec:
                    #     counter_word2vec.append(u)
                    #
                    # if v not in counter_word2vec:
                    #     counter_word2vec.append(v)

                    ## my_w2v_similarity
                    dG.edge[u][v]['w2vec'] = np.arccos(d['w2vec'])/math.pi
                    dG.edge[u][v]['w2vec'] = 1-dG.edge[u][v]['w2vec']
                    dG.edge[u][v]['weight'] = dG.edge[u][v]['w2vec']

                    ## attraction score
                    # d['w2vec'] = np.arccos(d['w2vec'])/math.pi
                    # f_u_v = float(dG.node[u]['count']*dG.node[v]['count'])/(d['w2vec']**2)
                    # dice = float(2*d['weight'])/(dG.node[u]['count']+dG.node[v]['count'])
                    # dG.edge[u][v]['weight'] = f_u_v * dice

                else:
                    dG.edge[u][v]['weight'] = 0.0001
                    # dG.edge[u][v]['weight'] = 1-dG.edge[u][v]['weight']
        '''
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

def init_environment(args):
    '''
    init_io = IO(
        args.work_dir,
        save_log=args.save_log,
        print_log=args.print_log)

    init_io.save_arg(args)
    '''
    gpus = visible_gpu(args.device)
    occupy_gpu(gpus)

    # return gpus, init_io
    return gpus

def load_model_(model, model_text, **model_args):
    model = model(**model_args)
    model_text += '\n\n' + str(model)
    return model, model_text

def load_model(args):
    model_text = ''
    model, model_text = load_model_(args.model, model_text, **(args.model_args))
    return model, model_text

def load_weights_(model, weights_path, ignore_weights=None):
    if ignore_weights is None:
        ignore_weights = []
    if isinstance(ignore_weights, str):
        ignore_weights = [ignore_weights]

    weights = torch.load(weights_path)
    weights = OrderedDict([[k.split('module.')[-1],
                            v.cpu()] for k, v in weights.items()])

    # filter weights
    for i in ignore_weights:
        ignore_name = list()
        for w in weights:
            if w.find(i) == 0:
                ignore_name.append(w)
        for n in ignore_name:
            weights.pop(n)

    try:
        model.load_state_dict(weights)
    except (KeyError, RuntimeError):
        state = model.state_dict()
        state.update(weights)
        model.load_state_dict(state)
    return model

def load_weights(args, model):
    if args.weights:
        model = load_weights_(model, args.weights, args.ignore_weights)
        return model

channel_count=1
num_class= 7
layout = 'coco'
args = parse_args()
args.model = 'net.st_gcn.Model'
args.model_args = {'in_channels': channel_count, 'num_class': num_class, 'edge_importance_weighting': True,
                   'graph_args': {'layout': layout, 'strategy': 'spatial'}}
args.use_gpu = True
#args.weights = '/home/lee/st-gcn/work_dir/recognition/kinetics_skeleton/ST_GCN/epoch50_model.pt'
#args.work_dir = './work_dir/recognition/kinetics_skeleton/ST_GCN'
gpus = init_environment(args)
model_pitcher, model_text = load_model(args)
model_pitcher = load_weights(args, model_pitcher)
stgcn_gpu = "cuda:1"
stgcn_pitcher = model_pitcher.to(stgcn_gpu)
stgcn_pitcher.eval()
stgcn_input = np.zeros((channel_count), dtype=np.float32)
data = torch.from_numpy(stgcn_input)
data = data.unsqueeze(0)
# data = data.float().to(stgcn_gpu).detach()
data = data.float().to(stgcn_gpu)
output = stgcn(data)
output_cpu = output.data.cpu().numpy()
label = np.argmax(output_cpu)