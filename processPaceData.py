#!/usr/bin/env python
# coding: utf-8

# In[72]:


import json
from datetime import datetime
import time
import math
import pandas as pd
import argparse
import os


# In[ ]:


#in order to align them every 10 or 30 seconds
#lets say tag1 starts at 6(unix time stamp) and tag2 starts at 10. Both have 1 measurement per #sec(sampling freq).
#Have to align tag1 with tag2 by taking the 5th measurement in tag 1
#First: we need to store the sampling frequences(hertz)
#Second:  I'm aligning everything with the largest unix time stamp. Take largest unix timestamp
#Third: For all the other timestamps, align them with the largest unix time stamp based on the sampling frequency.
#Store the offsets. 
#Fourth: All the tags should aligned and starting at the same data point. Now take every 10th or 30th data point


# In[ ]:


#example run
#python3 processPaceData.py -samplingTime 10 -directory directoryOnlyContainingData -outputDir outputDirectory


# In[ ]:


#assuming you have all avro files of the raw data in a directory


# In[45]:


#get largest timestamp and tag that contains that timestamp
def alignmentToLargestTimeStamp():
    largestTimestamp = 0
    for tag in data["rawData"]:
        try:
            largestTimestamp=max(data['rawData'][tag]['timestampStart'], largestTimestamp)
        except:
            #no timestamp
            pass

    #find the tag with the largest timestamp.
    toAlignTo=""
    for tag in data["rawData"]:
        try:
            if (data['rawData'][tag]['timestampStart']==largestTimestamp):
                toAlignTo=tag
        except:
            pass
    return largestTimestamp

#timestamp for gyroscope is 0??


# In[ ]:


def findOffsetsAndStepSizes(largestTimestamp):
    offsetOfTagDict={}

    #find offsets 
    for tag in data["rawData"]:
        try:
            timeDiffSec = largestTimestamp/1000000-(data["rawData"][tag]["timestampStart"])/1000000
            offsetOfTag=timeDiffSec * data["rawData"][tag]["samplingFrequency"]
            offsetOfTagDict[tag]=round(offsetOfTag)#round up or down?
        except:
            offsetOfTagDict[tag]="no timestampStart"

    #finding the step sizes given the offset. Start at offset, and then take the datapoints in increments of the step size
    stepSizeDict={}
    for tag in data["rawData"]:
        if offsetOfTagDict[tag]!='no timestampStart':
            stepSize=data["rawData"][tag]["samplingFrequency"]*samplingTime
            stepSizeDict[tag]=round(stepSize)
        else:
            stepSizeDict[tag]="no timestampStart"
    return offsetOfTagDict, stepSizeDict


# In[ ]:


def displayAlignedData(offsetOfTagDict, stepSizeDict, dataTagsDict):
    df = pd.DataFrame()
    for tag in data["rawData"]:
        #getting index of first data point
        if tag in dataTagsDict:
            for innerTag in dataTagsDict[tag]:
                lenData=len(data["rawData"][tag][innerTag])
                start = offsetOfTagDict[tag]
                currentStep=start

                dataList = []

                try:
                    dataList.append(data["rawData"][tag][innerTag][start])

                except:
                    continue
                i = offsetOfTagDict[tag]

                while i < (lenData-stepSizeDict[tag]-1):
                    currentStep = currentStep + stepSizeDict[tag]
                    i=i+ stepSizeDict[tag]
                    dataList.append(data["rawData"][tag][innerTag][currentStep])

                dataSeries=pd.Series(dataList)
                df[tag + " " + innerTag] = dataSeries

        else:
                continue
    return df
        


# In[ ]:


def addTimestamps(df, largestTimestamp, samplingTime):
    timeStampList=[]
    try:
        timeStampList.append(largestTimestamp)

    except:
        print("no timestamp")
    i=1
    nextTimeStamp=largestTimestamp+(samplingTime*1000000)
    timeStampList.append(nextTimeStamp)
    while i < df.shape[0]:
        i=i+1
        nextTimeStamp = nextTimeStamp + (samplingTime*1000000)
        timeStampList.append(nextTimeStamp)
    timeStampSeries=pd.Series(timeStampList)
    df["timeStamp"]=timeStampSeries
    timeStampSeries=timeStampSeries.apply(lambda x: datetime.utcfromtimestamp(x/1000000).strftime('%Y-%m-%d %H:%M:%S'))
    df["utcTimeStamp"]=timeStampSeries
    return df
        


# In[60]:



parser = argparse.ArgumentParser()


parser.add_argument("-samplingTime", required=True, type=int, metavar="samplingTime",help="samplingTime i.e 1 measurement every 10 seconds")
parser.add_argument("-directory", required=True,  metavar="paceDataDirectory", help="directory with avron files")
parser.add_argument("-outputDir", required=True,  metavar="outputDir", help="where csv files will be outputted")
args = parser.parse_args()
samplingTime=args.samplingTime
directory = args.directory
patientFiles=sorted(os.listdir(directory))
outputDir = args.outputDir

for file in patientFiles:
    jsonOutFile=directory+file+".json"
    os.system("avro-tools tojson" + " " + directory + "/" + file + " " + " > " + " " + jsonOutFile )
    f= open(jsonOutFile)
    data = json.load(f)
    largestTimeStamp=alignmentToLargestTimeStamp()
    offsetOfTagDict, stepSizeDict= findOffsetsAndStepSizes(largestTimeStamp)
    dataTagsDict = {"accelerometer" : ["x","y","z"], "gyroscope": ["x","y","z"], "eda": ["values"],
               "temperature": ["values"], "bvp":["values"], "steps": ["values"]}
    df =displayAlignedData(offsetOfTagDict, stepSizeDict, dataTagsDict)
    userID=directory.split("/")[8]
    df["userId"]=userID
    dfFinal = addTimestamps(df, largestTimeStamp, samplingTime)
    df.to_csv(os.path.join(outputDir, jsonOutFile + '.csv'))
    print(df)
    if file == patientFiles[0]:
        df.to_csv(os.path.join(outputDir, "all.csv"), index=False)#header is true
    else:
        df.to_csv(os.path.join(outputDir, "all.csv"), mode='a', index=False, header=False)
        print("hello")
    os.system("rm" + " " + jsonOutFile)


# In[85]:


list = [1,2,3,4,5,6,7]


# In[86]:


series = pd.Series(list)


# In[88]:


series.apply(lambda x:x*2)


# In[ ]:




