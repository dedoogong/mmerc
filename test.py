import csv
import glob
import pytext
import torch
import logging
logging.basicConfig(level=logging.INFO)

dataRootPath='/home/lee/Downloads/MELD.Raw/train/'
csvPath=dataRootPath+'train_sent_emo.csv'
dataGenerationFLAG=True # True , False
if dataGenerationFLAG:

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

'''
import mxnet as mx
from bert_embedding import BertEmbedding
bert_abstract = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
 Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
 As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. 
BERT is conceptually simple and empirically powerful. 
It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming human performance by 2.0%."""
sentences = bert_abstract.split('\n')
bert_embedding = BertEmbedding()
result = bert_embedding(sentences)

ctx = mx.gpu(0)
bert = BertEmbedding(ctx=ctx)

'''

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "[CLS] Um ok uh oh god um when you and uh Ross first started going out it was really hard for me um for many reasons which I'm not gonna bore you with now but um I just I see how happy he is you know and how good you guys are together and um Monica's always saying how nice you are and god I hate it when she's right [SEP]"
#"[CLS] Um, ok, uh, oh god, um, when you and uh Ross first started going out, it was really hard for me, um, for many reasons, which I'm not gonna bore you with now,[SEP] but um, I just, I see how happy he is, you know, and how good you guys are together, and um, Monica's always saying how nice you are, and god I hate it when she's right [SEP]"
#"[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
#masked_index = 8
#tokenized_text[masked_index] = '[MASK]'
#assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
#segments_ids = #[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_ids = [0]*77
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

################## BertModel

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    #a, b = model(tokens_tensor, segments_tensors)
    embeddings, encoded_layers, c = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
#assert len(encoded_layers) == 12
print(embeddings)
######################### BertForMaskedLM
'''
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')
'''
'''
# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'
'''
'''
import torch
import apex
input = torch.randn(20, 5, 10, 10)
# With Learnable Parameters
m = apex.normalization.FusedLayerNorm(input.size()[1:])
# Without Learnable Parameters
m = apex.normalization.FusedLayerNorm(input.size()[1:], elementwise_affine=False)
# Normalize over last two dimensions
m = apex.normalization.FusedLayerNorm([10, 10])
# Normalize over last dimension of size 10
m = apex.normalization.FusedLayerNorm(10)
# Activating the module
output = m(input)
'''