#!/usr/bin/env python
# coding: utf-8

# In[28]:


import json
from datetime import datetime
import time
import math
import pandas as pd
import argparse
import os
import pathlib


# In[29]:


def readAggregatedPaceData(inputDirectory, participantId):
    aggregatedFileDf = pd.DataFrame()
    inputDirectoryPath = pathlib.Path(inputDirectory)
    tmpFileDf = pd.DataFrame()
    tmpFileList = []
    for dir in inputDirectoryPath.iterdir():
        aggregatedFiles=[]
        list_dfs = []
        path = str(dir) + "/" + participantId + "/" + "digital_biomarkers/aggregated_per_minute"
        if(os.path.exists(path)): 
            print(path,len(list_dfs))
            for file in os.listdir(path):
                aggregatedFiles.append(os.path.join(path, file))
            for file in aggregatedFiles:
                data = pd.read_csv(file)
                list_dfs.append(data)
                #tmpFileDf=pd.concat(list_dfs, axis=1).T.drop_duplicates().T
                #tmpFileList.append(tmpFileDf)
            #tmpFileDf=pd.concat(map(pd.read_csv, aggregatedFiles), axis =1).T.drop_duplicates().T#issue probably here
            tmpFileDf=pd.concat(list_dfs, axis=1).T.drop_duplicates().T
            tmpFileDf=tmpFileDf.drop(columns=['missing_value_reason'])
            tmpFileList.append(tmpFileDf)
            #print(tmpFileDf)
            print(len(tmpFileDf))
        
    aggregatedFileDf = pd.concat(tmpFileList, axis =0)
#             if(aggregatedFileDf.empty):
#                 aggregatedFileDf=tmpFileDf
#                 aggregatedFileDf.reset_index(inplace=True, drop=True)
#             else:
#                 frames = [aggregatedFileDf, tmpFileDf]
#                 newaggregatedFileDf = pd.concat(frames, axis =0)
#                 tmpFileDf.reset_index(inplace=True, drop=True)
    #map applies pd.readcsv to every file in aggregatedFiles
    aggregatedFileDf = aggregatedFileDf.reset_index()
    return aggregatedFileDf
    #return tmpFileList


# In[30]:


#emp = readAggregatedPaceData("/Users/karlbrzoska/Downloads/1/1/participant_data/", "005-3YK3J151X6")


# In[31]:


def glucoseExtrapolation(dexcomInput, aggregatedFileDf):
    dexcom_participant=pd.read_csv(dexcomInput)


    dexcom_participant=dexcom_participant[dexcom_participant['Timestamp (YYYY-MM-DDThh:mm:ss)'].notnull()]
    dexcom_participant=dexcom_participant[dexcom_participant['Glucose Value (mg/dL)'].notnull()]

    #filter out na
    minimum = min(dexcom_participant['Timestamp (YYYY-MM-DDThh:mm:ss)'])
    maximum = max(dexcom_participant['Timestamp (YYYY-MM-DDThh:mm:ss)'])

    dexcom_participant["Glucose Value (mg/dL)"].replace({'High': 400},inplace=True)
    dexcom_participant["Glucose Value (mg/dL)"].replace({'Low': 100},inplace=True)

    df = pd.DataFrame(columns=["glucose", "timestamp_iso"])
    glucoseList =[]
    timeStampList=[]
    glucoseSeries=pd.Series()
    timeStampSeries=pd.Series()
    for index, row in aggregatedFileDf.iterrows():
        try:
            t = pd.to_datetime(row["timestamp_iso"])
            #returns dataframe where dexcom timestamp is less than, and then also greater than timestamp of current line
            dexcomdfLower=dexcom_participant[dexcom_participant['Timestamp (YYYY-MM-DDThh:mm:ss)']<row["timestamp_iso"]]
            
            dexcomdfUpper=dexcom_participant[dexcom_participant['Timestamp (YYYY-MM-DDThh:mm:ss)']>row["timestamp_iso"]]
            #looking for next value under current timestamp
            undert = pd.to_datetime(dexcomdfLower.iloc[len(dexcomdfLower)-1]["Timestamp (YYYY-MM-DDThh:mm:ss)"])
            overt = pd.to_datetime(dexcomdfUpper.iloc[0]["Timestamp (YYYY-MM-DDThh:mm:ss)"])
            underg=int(dexcomdfLower.iloc[len(dexcomdfLower)-1]["Glucose Value (mg/dL)"])
            overg=int(dexcomdfUpper.iloc[0]["Glucose Value (mg/dL)"])
            underDiff = abs(t.replace(tzinfo=None) -undert)
            underDiff= int(underDiff.total_seconds())
            overDiff = abs(t.replace(tzinfo=None) -overt)
            overDiff=int(overDiff.total_seconds())
            intTime =underDiff+overDiff

            glucose= underg * (overDiff/intTime) + overg * (underDiff/intTime)  
            timeStampList.append(row["timestamp_iso"])
            glucoseList.append(glucose)
        except:
            continue
    glucoseSeries=pd.Series(glucoseList)
    timeStampSeries=pd.Series(timeStampList)
    df["glucose"]=glucoseSeries
    df["timestamp_iso"]=timeStampSeries
    df = pd.merge(df, aggregatedFileDf, on='timestamp_iso')

    df.to_csv(combinedAggregatedFile)


# In[32]:


parser = argparse.ArgumentParser()


parser.add_argument("-directory", required=True,  metavar="aggregatedDataDirectory", help="directory with aggregated data")
parser.add_argument("-outputDir", required=True,  metavar="outputDir", help="where csv files will be outputted")
parser.add_argument("-dexcomInput", required=True,  metavar="dexcomFile", help="this is dexcom file, which is glucose data")
parser.add_argument("-participantId", required=True,  metavar="participantId", help="which participant being aligned")
parser.add_argument("-combinedAggregatedFile", required=True,  metavar="outputFile", help="file contained aligned aggregated files")
args = parser.parse_args()
directory = args.directory
outputDir = args.outputDir
dexcomInput = args.dexcomInput
combinedAggregatedFile = args.combinedAggregatedFile
participantId = args.participantId

aggregatedFileDf = readAggregatedPaceData(directory, participantId)
combinedAggregatedFile = outputDir + "/" + f"{participantId}.csv"
glucoseExtrapolation(dexcomInput, aggregatedFileDf)

# aggregatedFileDf = readAggregatedPaceData("/Users/karlbrzoska/Downloads/1/1/participant_data/", "005-3YK3J151X6")#readAggregatedPaceData(directory, participantId)
# combinedAggregatedFile = "/Users/karlbrzoska/Downloads/1/aggregatedOutput/005-3YK3J151X6.csv"#outputDir + "/" + f"{participantId}.csv"    
# dexcomInput = "/Users/karlbrzoska/Downloads/1/Dexcom/005-3YK3J151X6.csv"
# glucoseExtrapolation(dexcomInput, aggregatedFileDf)


# In[22]:


print(aggregatedFileDf)


# In[ ]:




