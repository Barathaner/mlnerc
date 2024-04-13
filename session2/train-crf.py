#!/usr/bin/env python3

import pycrfsuite
import sys
import json
import argparse

def instances(fi):
    xseq = []
    yseq = []
    for line in fi:
        line = line.strip('\n')
        if not line:
            yield xseq, yseq
            xseq = []
            yseq = []
            continue
        fields = line.split('\t')
        item = fields[5:]        
        xseq.append(item)
        yseq.append(fields[4])

def train_model(experiment_name, params, modelfile):
    trainer = pycrfsuite.Trainer()
    for xseq, yseq in instances(sys.stdin):
        trainer.append(xseq, yseq, 0)
    trainer.select('l2sgd', 'crf1d')
    for param, value in params.items():
        trainer.set(param, value)
    trainer.train(modelfile)
    print(f"Training {experiment_name} with following parameters:")
    for name in trainer.params():
        print(name, trainer.get(name), trainer.help(name), file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CRF model for a specific experiment.')
    parser.add_argument('experiment_name', help='The name of the experiment to run.')
    parser.add_argument('model_file', help='File path where the model should be saved.')
    args = parser.parse_args()

    with open('params_crf.json', 'r') as f:
        param_config = json.load(f)

    if args.experiment_name in param_config:
        params = param_config[args.experiment_name]
        train_model(args.experiment_name, params, args.model_file)
    else:
        sys.exit(f"Experiment {args.experiment_name} not found in the configuration.")
