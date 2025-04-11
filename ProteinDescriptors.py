# 提供了多种蛋白质序列特征描述符的计算方法，包括氨基酸组成(AAC)、二肽组成(DPC)、伪氨基酸组成(PseAAC)、物理化学性质描述符

import pandas as pd
import numpy as np 
from collections import Counter
import os, sys, re
import itertools
import math
import random
import pickle
# from preprocessing.data_util import read_fasta, assert_filetype, read_seq_file
from data_util import read_fasta, assert_filetype, read_seq_file
from rich.progress import track
import time 
        

def Count( seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum

    
def Count1(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code
        

def Rvalue(aa1, aa2, AADict, Matrix):
        return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def generatePropertyPairs(myPropertyName):
    pairs = []
    for i in range(len(myPropertyName)):
        for j in range(i + 1, len(myPropertyName)):
            pairs.append([myPropertyName[i], myPropertyName[j]])
            pairs.append([myPropertyName[j], myPropertyName[i]])
    return pairs
        
def toAAC(inFile,outCSV):
    T0=time.time()
    encoding_array = np.array([])
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    header=[]
    for aa in AA:
        header.append("Freq"+aa)
     
    print(header)
    encodings = []
    
    seq_pd=read_seq_file(inFile)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]


    
    for sequence in track(SEQ_,"Computing: "):
        count = Counter(sequence)
        for key in count:
            count[key] = count[key] / len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        
        #print(code)
        encodings.append(code)
        
    encoding_array = np.array(encodings, dtype=str)
    AAC_encoding=pd.DataFrame(encoding_array)
    AAC_encoding.columns=header
    
    AAC_encoding=pd.concat([CLASS_,AAC_encoding],axis=1)
    AAC_encoding.index=PID_
    print(AAC_encoding.shape)
    AAC_encoding.to_csv(outCSV)
    print("Getting Deep Representation Learning Features with UniRep is done.")
    
    

    
     
    print("AAC is DONE. AAC_encoding=",AAC_encoding.shape)
    print("it took %0.3f mins.\n"%((time.time()-T0)/60))
    return AAC_encoding
    
def toDPC(inFile,outCSV):
    encoding_array = np.array([])

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = ["Freq_"+aa1 + aa2 for aa1 in AA for aa2 in AA]
     
    
    seq_pd=read_seq_file(inFile)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]
    #print(header)

    DPC_list=[aa1 + aa2 for aa1 in AA for aa2 in AA]
    
    #print(DPC_list)
    
    

    
    for sequence in track(SEQ_,"computing:"):
        #name, sequence, Class = i[0], re.sub("-","",i[1]), str(i[2])
        #code = [name, Class]
        tmpCode = [0] * 400
        
        dpc_seq=[sequence[k]+sequence[k+1] for k in range(len(sequence)-1)]
        
        #print(dpc_seq)
        
        count = Counter(dpc_seq)
        
        for key in count:
            count[key] = count[key] / (len(sequence)-1)
            
        code =[]
        for idpc in DPC_list:
            code.append(count[idpc])
        
        #print(code)
        encodings.append(code)
        
        
        
         
     

    encoding_array = np.array(encodings, dtype=str)
    
    DPC_encoding=pd.DataFrame(encodings)
    
    DPC_encoding.columns=diPeptides

    DPC_encoding=pd.concat([CLASS_,DPC_encoding],axis=1)
    DPC_encoding.index=PID_
    print(DPC_encoding.shape)
    DPC_encoding.to_csv(outCSV)
    
    
    #DPC_encoding.to_csv(outCSV,index=False)
    #DPC_encoding.set_index("PID")
    #DPC_encoding.to_csv(outCSV)
    print("TODPC is DONE! DPC_encoding=",DPC_encoding.shape)
    
    return DPC_encoding
    
def toCTDC(inFasta,outCSV):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
        }
            
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
        }
    
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
                'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
                'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    
    for p in property:
        for g in range(1, len(groups) + 1):
            header.append(p + '.G' + str(g)+"_CTD.C")
            
            
    
    seq_pd=read_seq_file(inFasta)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]
    
    for sequence in track(SEQ_,"computing...."):
        code = []
        for p in property:
        
            c1 = Count(group1[p], sequence) / len(sequence)
            c2 = Count(group2[p], sequence) / len(sequence)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]
        encodings.append(code)

    encoding_array = np.array(encodings, dtype=str)
    
    CTDC_encoding=pd.DataFrame(encodings)
    
    CTDC_encoding.columns=header
    
    CTDC_encoding=pd.concat([CLASS_,CTDC_encoding],axis=1)
    CTDC_encoding.index=PID_
    print(CTDC_encoding.shape)
    CTDC_encoding.to_csv(outCSV)
  
    print("CTDC is DONE! CTDC_encoding=",CTDC_encoding.shape)
    
    return CTDC_encoding
    
def toCTDT(inFasta,outCSV):
    group1 = {
                'hydrophobicity_PRAM900101': 'RKEDQN',
                'hydrophobicity_ARGP820101': 'QSTNGDE',
                'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
                'hydrophobicity_PONP930101': 'KPDESNQT',
                'hydrophobicity_CASG920101': 'KDEQPSRNTG',
                'hydrophobicity_ENGD860101': 'RDKENQHYP',
                'hydrophobicity_FASG890101': 'KERSQD',
                'normwaalsvolume': 'GASTPDC',
                'polarity': 'LIFWCMVY',
                'polarizability': 'GASDT',
                'charge': 'KR',
                'secondarystruct': 'EALMQKRH',
                'solventaccess': 'ALFCGIVW'
            }
    group2 = {
                'hydrophobicity_PRAM900101': 'GASTPHY',
                'hydrophobicity_ARGP820101': 'RAHCKMV',
                'hydrophobicity_ZIMJ680101': 'HMCKV',
                'hydrophobicity_PONP930101': 'GRHA',
                'hydrophobicity_CASG920101': 'AHYMLV',
                'hydrophobicity_ENGD860101': 'SGTAW',
                'hydrophobicity_FASG890101': 'NTPG',
                'normwaalsvolume': 'NVEQIL',
                'polarity': 'PATGS',
                'polarizability': 'CPNVEQIL',
                'charge': 'ANCQGHILMFPSTWYV',
                'secondarystruct': 'VIYCWFT',
                'solventaccess': 'RKQEND'
            }
    group3 = {
                'hydrophobicity_PRAM900101': 'CLVIMFW',
                'hydrophobicity_ARGP820101': 'LYPFIW',
                'hydrophobicity_ZIMJ680101': 'LPFYI',
                'hydrophobicity_PONP930101': 'YMFWLCVI',
                'hydrophobicity_CASG920101': 'FIWC',
                'hydrophobicity_ENGD860101': 'CVLIMF',
                'hydrophobicity_FASG890101': 'AYHWVMFLIC',
                'normwaalsvolume': 'MHKFRYW',
                'polarity': 'HQRKNED',
                'polarizability': 'KMHFRYW',
                'charge': 'DE',
                'secondarystruct': 'GNPSD',
                'solventaccess': 'MSPTHY'
            }

    groups = [group1, group2, group3]
    property = (
                'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
                'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    
    for p in property:
        for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
            header.append(p + '.' + tr)
    
    seq_pd=read_seq_file(inFasta)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]

    for sequence in track(SEQ_,"computing..."):
       
        code = []
        aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
        
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                    
                if (pair[0] in group1[p] and pair[1] in group3[p]) or ( pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                    
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
                
            code = code + [c1221 /(1e-6+ len(aaPair)), c1331 / (1e-6+len(aaPair)), c2332 /(1e-6+ len(aaPair))]
        encodings.append(code)

    #encoding_array = np.array(encodings, dtype=str)
    
    CTDT_encoding=pd.DataFrame(encodings)
    
    CTDT_encoding.columns=header
    
    CTDT_encoding=pd.concat([CLASS_,CTDT_encoding],axis=1)
    CTDT_encoding.index=PID_
    print(CTDT_encoding.shape)
    CTDT_encoding.to_csv(outCSV)
   
    print("CTDT is DONE! CTDT_encoding=",CTDT_encoding.shape)        
     
    return CTDT_encoding

def toCTDD(inFasta,outCSV):

    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    for p in property:
        for g in ('1', '2', '3'):
            for d in ['0', '25', '50', '75', '100']:
                header.append(p + '.' + g + '.residue' + d)
    
    
    seq_pd=read_seq_file(inFasta)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]

    for sequence in track(SEQ_,"computing..."):

        #name, sequence, Class = i[0], re.sub('-', '', i[1]), str(i[2])
        code = []
        for p in property:
            code = code +  Count1(group1[p], sequence) +  Count1(group2[p], sequence) +  Count1(
                group3[p], sequence)
                
        encodings.append(code)

    CTDD_encoding=pd.DataFrame(encodings)
    
    CTDD_encoding.columns=header
    
    CTDD_encoding=pd.concat([CLASS_,CTDD_encoding],axis=1)
    CTDD_encoding.index=PID_
    print(CTDD_encoding.shape)
    CTDD_encoding.to_csv(outCSV)
   
    print("CTDD is DONE! CTDD_encoding=",CTDD_encoding.shape) 
    
    return CTDD_encoding
    
def toCTD(inFasta,outCSV):
    
    ctdc=toCTDC(inFasta,outCSV)
    #print(ctdc.head())
    ctdt=toCTDT(inFasta,outCSV)
    #print(ctdt.head())
    ctdd=toCTDD(inFasta,outCSV)
    #print(ctdd.head())
    
    ctd=pd.concat([ctdc,ctdt.iloc[:,1:],ctdd.iloc[:,1:]],axis=1)
    ctd.to_csv(outCSV)
    print("CTD is DONE@ CTD_encoding=",ctd.shape)
    
    #print("CTD is DONE@ CTD_encoding=",ctd.shape)
    return ctd
    

def toPAAC(inFasta,outCSV,lambdaValue = 2,w = 0.05):
 
    lambdaValue = lambdaValue
    w = w
    
    dataFile = './pyModels/artfeat/util_data/PAAC.txt'
    
    with open(dataFile) as f:
        records = f.readlines()
        
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    
    for i in range(len(AA)):
        AADict[AA[i]] = i
        
    AAProperty = []
    AAPropertyNames = []
    
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])
    
    encodings = []
    header = []
    
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
     
    seq_pd=read_seq_file(inFasta)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]
    
    for sequence in track(SEQ_,"Computing..."):
        #name, sequence, Class = i[0], re.sub('-', '', i[1]), str(i[2])
        code = []
        
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in
                    range(len(sequence) - n)]) / (
                        len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
        
    
    PAAC_encoding=pd.DataFrame(encodings)
    
    PAAC_encoding.columns=header
    PAAC_encoding=pd.concat([CLASS_,PAAC_encoding],axis=1)
    PAAC_encoding.index=PID_
    print(PAAC_encoding.shape)
    PAAC_encoding.to_csv(outCSV)
    
    #PAAC_encoding.to_csv(outCSV,index=False)
    #PAAC_encoding.set_index("PID")
    #PAAC_encoding.to_csv(outCSV)
    print("PAAC is DONE! PAAC_encoding=",PAAC_encoding.shape) 
    
    return PAAC_encoding  
    

def toAPAAC(inFasta,outCSV,lambdaValue = 2,weight = 0.05):
 
    lambdaValue =  lambdaValue 
    
    w = weight
    dataFile= './pyModels/artfeat/util_data/PAAC.txt'
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records) - 1):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = []
    for i in AA:
        header.append('Pc1.' + i)
    for j in range(1, lambdaValue + 1):
        for i in AAPropertyNames:
            header.append('Pc2.' + i + '.' + str(j))
    
    
    seq_pd=read_seq_file(inFasta)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]
    
    for sequence in  track(SEQ_,"computing..."):
        #name, sequence, Class = i[0], re.sub('-', '', i[1]), str(i[2])
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(
                    sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                        range(len(sequence) - n)]) / (len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)

        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [w * value / (1 + w * sum(theta)) for value in theta]
        encodings.append(code)
    
    APAAC_encoding=pd.DataFrame(encodings)
    
    APAAC_encoding.columns=header
    
    APAAC_encoding=pd.concat([CLASS_,APAAC_encoding],axis=1)
    APAAC_encoding.index=PID_
    print(APAAC_encoding.shape)
    APAAC_encoding.to_csv(outCSV)
    print("APAAC is DONE! APAAC_encoding=",APAAC_encoding.shape) 
    
    return APAAC_encoding

''' KNN descriptor '''
def Sim( a, b):
    blosum62 = [
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, 0],  # A
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, 0],  # N
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, 0],  # D
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, 0],  # Q
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, 0],  # E
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, 0],  # G
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, 0],  # H
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, 0],  # I
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, 0],  # L
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, 0],  # K
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, 0],  # M
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, 0],  # F
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, 0],  # P
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, 0],  # S
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, 0],  # T
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, 0],  # W
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, 0],  # Y
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, 0],  # V
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0],  # -
    ]
    AA = 'ARNDCQEGHILKMFPSTWYV-'
    myDict = {}
    for i in range(len(AA)):
        myDict[AA[i]] = i
        maxValue, minValue = 11, -4
    return (blosum62[myDict[a]][myDict[b]] - minValue) / (maxValue - minValue)

def CalculateDistance(sequence1, sequence2):
    seq1=sequence1
    seq2=sequence2
    if len(seq1)> len(seq2):
       for i in range(len(seq1)-len(seq2)):
            seq2=seq2+"-"
       print("sequence not equal and filled with -")
    elif   len(seq1)< len(seq2):
        for i in range(len(seq1)-len(seq2)):
            seq1=seq1+"-"
        print("sequence not equal and filled with -")
        
         
    distance = 1 - sum([Sim(seq1[i], seq2[i]) for i in range(len(seq1))]) / len(seq1)
    return distance

def CalculateContent( myDistance, j, myClassSets):
    content = []
    myDict = {}
    for i in myClassSets:
        myDict[i] = 0
    for i in range(j):
        myDict[myDistance[i][0]] = myDict[myDistance[i][0]] + 1
    for i in myClassSets:
        content.append(myDict[myClassSets[i]] / j)
    return content

def toKNN(inFasta,outCSV):


    encoding_array = np.array([])

    
    topK_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]

    training_data = []
    training_Class = {}

    seq_pd=read_seq_file(inFasta)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]
    for i in SEQ_:
         
        training_data.append(i)
        training_Class[i[0]] = int(i[2])
        
    tmp_Class_sets = list(set(training_Class.values()))

    topK_numbers = []
    for i in topK_values:
        topK_numbers.append(math.ceil(len(training_data) * i))

    # calculate pair distance
    distance_dict = {}
    for i in track(range(len(SEQ_)),"computing..."):
        sequence_1 = SEQ_[i] 
        for j in range(i+1, len(SEQ_)):
            sequence_2 =  SEQ_[j] 
             
            distance_dict[':'.join(sorted([name_seq1, name_seq2]))] = CalculateDistance(sequence_1, sequence_2)

    encodings = []
    header = []
    for k in topK_numbers:
        for l in tmp_Class_sets:
            header.append('Top' + str(k) + '.Class' + str(l))
     

    for i in track(SEQ_,"computing..."):
       
        code = []
        tmp_distance_list = []
        for j in range(len(training_data)):
            if name != training_data[j][0]:
                tmp_distance_list.append([int(training_data[j][2]), distance_dict.get(':'.join(sorted([name, training_data[j][0]])), 1)])

        tmp_distance_list = np.array(tmp_distance_list)
        tmp_distance_list = tmp_distance_list[np.lexsort(tmp_distance_list.T)]

        for j in topK_numbers:
            code += CalculateContent(tmp_distance_list, j, tmp_Class_sets)
        encodings.append(code)
    
    KNN_encoding=pd.DataFrame(encodings)
    
    KNN_encoding.columns=header
    KNN_encoding=pd.concat([CLASS_,KNN_encoding],axis=1)
    KNN_encoding.index=PID_
    print(APAAC_encoding.shape)
    KNN_encoding.to_csv(outCSV)



    
    #KNN_encoding.to_csv(outCSV,index=False)
    #KNN_encoding.set_index("PID")
    #KNN_encoding.to_csv(outCSV)
   
    print("KNN is DONE! KNN_encoding=",KNN_encoding.shape) 
    
    return KNN_encoding
     
''' end Protein KNN descriptor '''

def toAutoCrossCov(inFasta,outCSV,nlag=2):
 
    aaindex=pd.read_csv('./util_data/AAindex.txt',sep='\t',header=0)
    print(aaindex)

    property_name = aaindex["AccNo"]
    print(len(property_name))
    print(property_name[3:10])
    
     

    nlag = nlag

     

     
    data_file =  './util_data/AAindex.data'
    with open(data_file, 'rb') as handle:
        property_dict = pickle.load(handle)
        
    

    for p_name in property_name:
        tmp = np.array(property_dict[p_name], dtype=float)
        pmean = np.average(tmp)
        pstd = np.std(tmp)
        property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

    AA = 'ARNDCQEGHILKMFPSTWYV'
    AA_order_dict = {}
    for i in range(len(AA)):
        AA_order_dict[AA[i]] = i

    property_pairs = generatePropertyPairs(property_name)

    encodings = []
    header = ['PID', 'Class']
    for p_name in property_name:
        for i in range(nlag):
            header.append('%s.lag%s' % (p_name, i + 1))
    header += [p[0] + '_' + p[1] + '_lag.' + str(lag) for p in property_pairs for lag in range(1, nlag + 1)]
    
    
    fasta_list = read_fasta(inFasta)

    for i in fasta_list:
        name, sequence, Class = i[0], re.sub('-', '', i[1]), str(i[2])
        code = [name, Class]

        L = len(sequence)
        for p_name in property_name:
            xmean = sum([property_dict[p_name][AA_order_dict[aa]] for aa in sequence]) / L
            for lag in range(1, nlag + 1):
                ac = 0
                try:
                    ac = sum([(property_dict[p_name][AA_order_dict[sequence[j]]] - xmean) * (property_dict[p_name][AA_order_dict[sequence[j+lag]]] - xmean) for j in range(L - lag)])/(L-lag)
                except Exception as e:
                    ac = 0
                code.append(ac)
        for pair in property_pairs:
            mean_p1 = sum([property_dict[pair[0]][AA_order_dict[aa]] for aa in sequence]) / L
            mean_p2 = sum([property_dict[pair[1]][AA_order_dict[aa]] for aa in sequence]) / L
            for lag in range(1, nlag + 1):
                cc = 0
                try:
                    cc = sum([(property_dict[pair[0]][AA_order_dict[sequence[j]]] - mean_p1) * (
                                property_dict[pair[1]][AA_order_dict[sequence[j + lag]]] - mean_p2) for j in
                            range(L - lag)]) / (L - lag)
                except Exception as e:
                    cc = 0
                code.append(cc)
        encodings.append(code)
        
    AutoCrossCov_encoding=pd.DataFrame(encodings)
    
    AutoCrossCov_encoding.columns=header
    
    AutoCrossCov_encoding.to_csv(outCSV,index=False)
    #AutoCrossCov_encoding.set_index("PID")
    #AutoCrossCov_encoding.to_csv(outCSV)
   
    print(" AutoCrossCov is DONE!  AutoCrossCov_encoding=",AutoCrossCov_encoding.shape) 
    
    return AutoCrossCov_encoding
     
def toASDC(inFasta,outCSV):
     
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = []
    header += ["ASDC_"+aa1 + aa2 for aa1 in AA for aa2 in AA]
    
    seq_pd=read_seq_file(inFasta)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]

    for sequence in track(SEQ_,"computing..."):
        #name, sequence, Class = i[0], re.sub('-', '', i[1]), str(i[2])
        code = []
        sum = 0
        pair_dict = {}
        for pair in aaPairs:
            pair_dict[pair] = 0
        for j in range(len(sequence)):
            for k in range(j + 1, len(sequence)):
                if sequence[j] in AA and sequence[k] in AA:
                    pair_dict[sequence[j] + sequence[k]] += 1
                    sum += 1
        for pair in aaPairs:
            code.append(pair_dict[pair] /(1e-6+ sum))
        encodings.append(code)
    
    ASDC_encoding=pd.DataFrame(encodings)
    
    ASDC_encoding.columns=header
    ASDC_encoding=pd.concat([CLASS_,ASDC_encoding],axis=1)
    ASDC_encoding.index=PID_
    print(ASDC_encoding.shape)
    ASDC_encoding.to_csv(outCSV)

     
    #ASDC_encoding.to_csv(outCSV,index=False)
    #ASDC_encoding.set_index("PID")
    #ASDC_encoding.to_csv(outCSV)
    
   
    print(" ASDC is DONE!  ASDC_encoding=",ASDC_encoding.shape) 
    
    return ASDC_encoding
     
     
def toBLOSUM62(inFasta,outCSV):
 
    
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }
    encodings = []
    header = []
    
    AA='ACDEFGHIKLMNPQRSTVWY-'
    
    
   
    
    for i in range(21*20):
        header.append('blosum62.F' + str(i+1))
    
    
    seq_pd=read_seq_file(inFasta)
    SEQ_=seq_pd["Sequence"]
     
    CLASS_=seq_pd["Target"]
    PID_=seq_pd["PID"]
   

    for sequence in track(SEQ_,"computing..."):
        #name, sequence, Class = i[0], i[1], i[2]
        code = []
        count=Counter(sequence)
        #print(count)
       # print(count.keys())
        for aa in AA:
            if aa in count.keys():
                code = code + [i*count[aa] for i in blosum62[aa]]
            else:
                code = code +  blosum62['-']
            
        encodings.append(code)
    
    
    blosum62_encoding=pd.DataFrame(encodings)
    print(blosum62_encoding)
    

    blosum62_encoding.columns=header
    blosum62_encoding=pd.concat([CLASS_,blosum62_encoding],axis=1)
    blosum62_encoding.index=PID_
    print(blosum62_encoding.shape)
    blosum62_encoding.to_csv(outCSV)
    
    #blosum62_encoding.to_csv(outCSV,index=False)
    #blosum62_encoding.set_index("PID")
    #blosum62_encoding.to_csv(outCSV)
   
    print(" blosum62 is DONE!  blosum62_encoding=",blosum62_encoding.shape) 
    
    return blosum62_encoding

     
    
