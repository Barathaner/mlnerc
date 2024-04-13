# Define BASEDIR relative to the script location
$BASEDIR = "C:\Users\User\mlnerc-1"

# Convert datasets to feature vectors
Write-Host "Extracting features..."
python .\extract-features.py $BASEDIR\DDI\data\train\ > .\train.feat
python .\extract-features.py $BASEDIR\DDI\data\devel\ > .\devel.feat

# Train CRF model
Write-Host "Training CRF model..."
Get-Content .\train.feat | python .\train-crf.py .\model.crf

# Run CRF model
Write-Host "Running CRF model..."
Get-Content .\devel.feat | python .\predict.py .\model.crf > .\devel-CRF.out

# Evaluate CRF results
Write-Host "Evaluating CRF results..."
python .\evaluator.py NER $BASEDIR\DDI\data\devel .\devel-CRF.out > .\devel-CRF.stats

# Extract Classification Features
Get-Content .\train.feat | Where-Object { $_ -match "\S" } | ForEach-Object { $_ -replace "^[^\t]*\t[^\t]*\t[^\t]*\t[^\t]*\t", "" } | Set-Content .\train.clf.feat

# Train Naive Bayes model
Write-Host "Training Naive Bayes model..."
Get-Content .\train.clf.feat | python .\train-sklearn.py .\model.joblib .\vectorizer.joblib

# Run Naive Bayes model
Write-Host "Running Naive Bayes model..."
Get-Content .\devel.feat | python .\predict-sklearn.py .\model.joblib .\vectorizer.joblib > .\devel-NB.out

# Evaluate Naive Bayes results
Write-Host "Evaluating Naive Bayes results..."
python .\evaluator.py NER $BASEDIR\DDI\data\devel .\devel-NB.out > .\devel-NB.stats
