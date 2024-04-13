Befehle:
CRF:

1. Features extrahieren
python ./extract-features.py ../DDI/data/devel/ > devel.feat
2. trainieren
Get-Content devel.feat | python .\train-crf.py .\model.crf  
3. predicten
Get-Content devel.feat | python predict.py model.crf | Out-File devel.out
4. eval
python evaluator.py NER ../DDI/data/devel devel.out

NAIVE BAYES:

Befehle:
1. Features extrahieren
python ./extract-features.py ../DDI/data/devel/ > devel.feat
2. trainieren
Get-Content devel.feat | python .\train-sklearn.py model.joblib vectorizer.joblib
3. predicten
Get-Content test.feat | python .\predict-sklearn.py .\model.joblib .\vectorizer.joblib > Out-File test.out4. eval
python evaluator.py NER ../DDI/data/devel devel.out