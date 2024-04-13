# Define BASEDIR relative to the script location
$BASEDIR = "C:\Users\User\mlnerc-1"

# Convert datasets to feature vectors
Write-Host "Extracting features..."
python .\extract-features.py $BASEDIR\DDI\data\train\ > .\train.feat
python .\extract-features.py $BASEDIR\DDI\data\devel\ > .\devel.feat
# Define the path to the parameter configuration file
$paramConfigFile = ".\params_crf.json"

# Load the configuration for experiments
$experiments = Get-Content $paramConfigFile | ConvertFrom-Json

foreach ($experiment in $experiments.PSObject.Properties) {
    $expName = $experiment.Name
    $modelFile = ".\${expName}_model.crf"
    $outputFile = ".\${expName}-CRF.out"
    $statsFile = ".\${expName}-CRF.stats"

    # Train CRF model for each experiment
    Write-Host "Training CRF model for $expName..."
    Get-Content .\train.feat | python .\train-crf.py $expName $modelFile

    # Run CRF model for each experiment
    Write-Host "Running CRF model for $expName..."
    Get-Content .\devel.feat | python .\predict.py $modelFile > $outputFile

    # Evaluate CRF results for each experiment
    Write-Host "Evaluating CRF results for $expName..."
    python .\evaluator.py NER $BASEDIR\DDI\data\devel $outputFile > $statsFile
}

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

