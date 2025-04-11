import pandas as pd
import numpy as np
import joblib
from avp_data_util import  toBLOSUM62_NoTarget,get_ee_feats
from diffEntropy import entropy_with_metrics
#加载训练好的模型
std_scaler=joblib.load("AVP.StandardScaler.joblib")

vote_rf_model=joblib.load("VOTErf01.SMOTE.PlusBLOSUM_UMAP_EF2.SMOTE.WHTAVP_BLOSUM62.F5F470_VOTE1modelWith145Features.joblib")

blosum62_to_uamp=joblib.load("BLOSUM62toUAMP.fitModel.joblib")
#获取最佳特征组合
topFeats=pd.read_csv("topFeats.csv",header=0)

def getFeats(inFasta):
    blosum62_feats=toBLOSUM62_NoTarget(inFasta)
    umap_feats=blosum62_to_uamp.transform(blosum62_feats)
    umap_feats=pd.DataFrame(umap_feats)
    umap_feats.columns=["UMAP_F" +str(i) for i in range(1,umap_feats.shape[1]+1)]
    ee_feats=get_ee_feats(blosum62_feats,umap_feats)
    combined_feats = pd.concat([
        blosum62_feats.reset_index(drop=True),
        umap_feats.reset_index(drop=True),
        ee_feats.reset_index(drop=True)
    ], axis=1)
    print(combined_feats.shape)
    print(combined_feats.head())
    return combined_feats
feats=getFeats("AVP_WHT_lenLessThan50AA.i30.test4kfold.fasta")

feats=feats[topFeats.columns]
feats_std=std_scaler.transform(feats)
pred_labels=vote_rf_model.predict(feats_std[:,:145])
pred_probas=vote_rf_model.predict_proba(feats_std[:,:145])
pred_results=pd.DataFrame()

pred_results["pred_label"]=pred_labels
pred_results["pred_proba"]=pred_probas[:,1]
pred_results.to_csv("测试_预测结果.csv")