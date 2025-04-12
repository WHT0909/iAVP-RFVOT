import pandas as pd
import umap 
import joblib
from diffEntropy import entropy_with_metrics

def getUmapFeat(trainCSV,testCSV,umap_dim=50,out="UMAP_feat",metric="yule"):
    trainData=pd.read_csv(trainCSV,header=0)
    testData=pd.read_csv(testCSV,header=0)
    print("TrainData:",trainData.shape)
    print("TestData:",testData.shape)
    myUMAP=umap.UMAP(n_components=umap_dim,metric=metric)
    train_umap_data=myUMAP.fit_transform(trainData.iloc[:,2:])
    print(train_umap_data.shape)
    test_umap_data=myUMAP.transform(testData.iloc[:,2:])
    print(test_umap_data.shape)
    train_umap_data_pd=pd.DataFrame(train_umap_data,columns=[metric+"UMAP_F"+str(i+1) for i in range(0,umap_dim)])
    test_umap_data_pd=pd.DataFrame(test_umap_data,columns=["metric+UMAP_F"+str(i+1) for i in range(0,umap_dim)])
    train_umap_data_pd=pd.concat([trainData.iloc[:,:2],train_umap_data_pd],axis=1)
    test_umap_data_pd=pd.concat([testData.iloc[:,:2],test_umap_data_pd],axis=1)
    train_umap_data_pd.to_csv(out+".UmapFeat.Train.csv",index=False)
    test_umap_data_pd.to_csv(out+".UmapFeat.Test.csv",index=False)
    joblib.dump(myUMAP,out+".UAMP.fitModel.joblib")
    return train_umap_data_pd,test_umap_data_pd


def getUmapEntropyFeat(trainCSV,testCSV,umap_dim=50,metric="yule",out="UMAP_DiffEntropy_Feat"):
    trainData=pd.read_csv(trainCSV,header=0)
    testData=pd.read_csv(testCSV,header=0)
    train_entropy=[]
    test_entropy=[]
    print("$%"*30)
    print("获取原始特征的微分熵")

    for i in range(trainData.shape[0]):
        ee=entropy_with_metrics(trainData.iloc[i,2:].values.reshape(-1,1))
        train_entropy.append(ee)
    for i in range(testData.shape[0]):
        ee=entropy_with_metrics(testData.iloc[i,2:].values.reshape(-1,1))
        test_entropy.append(ee)
    train_data_entropy=trainData.iloc[:,:2]
    test_data_entropy=testData.iloc[:,:2]
    train_data_entropy["diffEntropy_F0"]=train_entropy
    test_data_entropy["diffEntropy_F0"]=test_entropy
    print("Train Entropy :",train_data_entropy)
    print("Test Entropy ：", test_data_entropy)
    train_data_entropy.to_csv(out+".Train.diffEntropyF0.csv",index=False)
    test_data_entropy.to_csv(out+".Test.diffEntropyF0.csv",index=False)


    print("#$"*20)
    print("UMAP降维处理")
    train_umap_entropy=[]
    test_umap_entropy=[]
    train_umap,test_umap=getUmapFeat(trainCSV,testCSV,umap_dim=umap_dim,out=out,metric=metric)

    for i in range(train_umap.shape[0]):
        ee=entropy_with_metrics(train_umap.iloc[i,2:].values.reshape(-1,1))
        train_umap_entropy.append(ee)
    for i in range(test_umap.shape[0]):
        ee=entropy_with_metrics(test_umap.iloc[i,2:].values.reshape(-1,1))
        test_umap_entropy.append(ee)

    train_data_entropy["UMAP_diffEntropy_F0"]=train_umap_entropy
    test_data_entropy["UMAP_diffEntropy_F0"]=test_umap_entropy
    train_data_entropy.to_csv(out+".Train.UMAP_diffEntropyF0.csv",index=False)
    test_data_entropy.to_csv(out+".Test.UMAP_diffEntropyF0.csv",index=False)

    print("@#"*20)
    print("构建UMAP+Entropy特征,维度UMAP_dim+2")
    print(train_umap.shape,test_umap.shape)
    train_umap["Entropy_F0"]=train_entropy
    train_umap["UMAP_diffEntropy_F0"]=train_umap_entropy
    test_umap["Entropy_F0"]=test_entropy
    test_umap["UMAP_diffEntropy_F0"]=test_umap_entropy
    print(train_umap.shape,test_umap.shape)
    print(train_umap.head())
    print(test_umap.head())
    train_umap.to_csv(out+".Train.UMAP_diffEntropyF0F1.csv",index=False)
    test_umap.to_csv(out+".Test.UMAP_diffEntropyF0F1.csv",index=False)
    print("原始特征+UMAP+Entropy"+"^*"*30)
    print(trainData.shape,testData.shape)
    trainData=pd.concat([trainData,train_umap.iloc[:,2:]],axis=1)
    testData=pd.concat([testData,test_umap.iloc[:,2:]],axis=1)
    trainData.to_csv(out+".PlusTrain.UMAP_diffEntropyF0F1.csv",index=False)
    testData.to_csv(out+".PlusTest.UMAP_diffEntropyF0F1.csv",index=False)
    print(trainData.shape,testData.shape)
    print(testData.head(3)) 
    print("GOOD LUCK!!!")
