# iAVP-RFVOT

## Project Introduction
This project is a machine learning-based antiviral peptide prediction system using random forest voting model combined with BLOSUM62 and UMAP feature extraction methods.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Prepare input FASTA file (example: AVP_WHT_lenLessThan50AA.i30.test4kfold.fasta)
2. Run prediction script:
```bash
python predict.py
```

## File Description
- `predict.py`: Main prediction script
- `avp_data_util.py`: Data processing utility
- `umapFeat.py`: UMAP feature extraction tool
- `*.joblib`: Pre-trained model files
- `topFeats.csv`: Optimal feature combinations

## Output Description
Prediction results will be saved in `result.csv`, containing two columns:
- `pred_label`: Prediction label (0/1)
- `pred_proba`: Prediction probability value

## Notes
Please ensure all model files (.joblib) and input FASTA files are in the same directory as the scripts.
