from diffEntropy import entropy_with_metrics
from collections import Counter
import pandas as pd
def read_fasta(fasta_file):
    """读取FASTA文件，返回包含PID和Sequence的DataFrame，用于处理没有Target的Fasta文件"""
    sequences = []
    pids = []
    
    with open(fasta_file, 'r') as f:
        current_pid = ""
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_pid:  # 保存上一个序列
                    sequences.append(current_seq)
                    pids.append(current_pid)
                    current_seq = ""
                current_pid = line[1:]  # 去掉>符号
            else:
                current_seq += line
        # 添加最后一个序列
        if current_pid:
            sequences.append(current_seq)
            pids.append(current_pid)
    
    return pd.DataFrame({"PID": pids, "Sequence": sequences})

def toBLOSUM62_NoTarget(inFasta):
    """生成不含Target列的BLOSUM62特征"""
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    
    AA = 'ACDEFGHIKLMNPQRSTVWY-'
    header = ['blosum62.F' + str(i+1) for i in range(21*20)]
    encodings = []
    
    # 读取FASTA文件
    seq_df = read_fasta(inFasta)
    print(f"读取到 {len(seq_df)} 条序列")
    
    # 生成BLOSUM62特征
    for seq in seq_df["Sequence"]:
        code = []
        count = Counter(seq)
        for aa in AA:
            if aa in count:
                code.extend([score * count[aa] for score in blosum62[aa]])
            else:
                code.extend(blosum62['-'])
        encodings.append(code)
    
    # 创建DataFrame
    blosum_df = pd.DataFrame(encodings, columns=header)
    blosum_df.index= seq_df["PID"]  # 设置PID为索引
    
    # 保存结果
    #blosum_df.to_csv(outCSV)
    #print(f"BLOSUM62特征生成完成，形状: {blosum_df.shape}，已保存到 {outCSV}")
    
    return blosum_df

def get_ee_feats(blosum62_feats,umap_feats):
	print(blosum62_feats.shape,umap_feats.shape)
	ee_feats_blosum62=[]
	ee_feats_umap=[]
	
	for i in range(blosum62_feats.shape[0]):
		ee=entropy_with_metrics(blosum62_feats.iloc[i,2:].values.reshape(-1,1))
		ee_feats_blosum62.append(ee)

	for i in range(umap_feats.shape[0]):
		ee=entropy_with_metrics(umap_feats.iloc[i,2:].values.reshape(-1,1))
		ee_feats_umap.append(ee)
	ee_feats=pd.DataFrame()
	ee_feats["Entropy_F0"]=ee_feats_blosum62
	ee_feats["UMAP_diffEntropy_F0"]=ee_feats_umap
	print(ee_feats.shape)
	print(ee_feats.head())
	return ee_feats
