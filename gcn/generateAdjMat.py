import csv
import glob
import numpy as np
import torch
import logging
import cv2
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
#9971 9988
print('----------------------------------------------')
speakerCount=len(speakersNetList)
uttersCount=len(sentimentIDs)

nodeCount=len(sentimentIDs)+len(speakersNetList)
adjMat=np.zeros((nodeCount,nodeCount),dtype=np.float32)

iMat = np.identity(speakerCount, dtype = np.float32)
adjMat[uttersCount:,uttersCount:]=iMat

dialogueSizeList=[]
idx=0
size=0
for d in dialogueIDs:
    if d == idx:
        size+=1
    else:
        dialogueSizeList.append(size)
        size=1
        idx+=1

offset=0
for i, d in enumerate(dialogueSizeList):
    dialogueSubMat = np.ones((d, d), dtype=np.float32)
    adjMat[offset:offset+d,offset:offset+d]=dialogueSubMat
    offset+=d
    #print(offset)

d = len(sentimentIDs)-offset
dialogueSubMat = np.ones((d, d), dtype=np.float32)
adjMat[offset:offset+d, offset:offset+d] = dialogueSubMat
offset += d

utter_speaker_Mat = np.zeros((uttersCount,speakerCount),dtype=np.float32) # rows x columns
for i,_ in enumerate(speakerIDs):
    utter_speaker_Mat[i,speakerIDs[i]]=1

adjMat[:uttersCount, -speakerCount:]=utter_speaker_Mat
adjMat[uttersCount:, :uttersCount]=np.transpose(utter_speaker_Mat)
cv2.imwrite('./adjMat.jpg',adjMat*255)
#cv2.waitKey(0)
#print("offset")