# SMELLCPS Translation Server

## Pre- and postprocessing of data
### Preprocessing
TODO expand
- ASCII check
- additional formatting checks should be included as if-else statements at line XXXXXX of file XXXXXX
- timeout not currently implemented
- after validity checks, remapping
- all tokens except +, -, *, /, -1 (no space), (), [], {} are considered variable names and canonicalized accordingly.
  - should expand list at line XXXXX of app.py if we need to generalize to other inputs
- space tokenization, no BPE or subword segmentation applied
- 
### Postprocessing
TODO expand
- unremapping
- tokens joined with spaces and returned as string

## How to use: Browser UI
- URL: TODO
- Type SYM expressions in the left, hit go, get a MATH expression on the right

## How to use: JSON API
1) Send a HTTP POST request to TODO URL:
```commandline
 curl --data "source=<SYM EXPR>" http://TODO:6060/translate
```
2) The JSON response will be:
```commandline
{
  "source": [
    "<SYM EXPR>",
  ],
  "translation": [
    "<MATH EXPR>",
  ]
}
```
You can also send multiple source expressions in one request.
3) POST is the preferred method, but GET requests are also accepted.

## Organization of code
TODO

## About the translation model
TODO

## How to train a new model
1. In the [math_to_symbexpr_map_generation/experiments](https://github.com/usc-isi-bass/math_to_symbexpr_map_generation/tree/exp/seq2seq_math/experiments) directory, make a new directory and copy over the following files from 15_remap_infix:
   - preprocess.job
   - run_15.job (rename to match experiment number)
   - evaluate_valid.job
   - evaluate_test.job
   - all files in data/
   - 
2. Preprocess and binarize data. The arguments in preprocess.job should work without changes.
```commandline
sbatch preprocess.job
```

3. Train a new model. Update run_xx.job to choose your hyperparameters. The ones in run_15.job are from our current best model, so it's a good starting point.
```commandline
sbatch run_xx.job
```

4. Check model performance using Tensorboard, the job log in run.out, and the evaluation scripts. 
Empirically, these models seem to require a LOT of training to perform well - >7M training steps. 

## How to upload a new model
To run a fairseq model on the server, you need the model checkpoint (a .pt file, usually `checkpoints/checkpoint_best.pt` or `checkpoints/checkpoint_last.pt`) and the binarized data associated with the model.
The binarized data is usually the whole `data-bin/` directory from inside your experiment directory.
- if you upload a new checkpoint of the same model (i.e. trained on the same `data-bin/` from the same preprocessing job), you do not need to reupload binarized data
- if you upload a new model (not a checkpoint of the previous model), you should also upload its binarized data.
Model files are usually >100MB. If you want to push/pull them with Github, you'll need Git LFS. 
After uploading your checkpoint, make sure the filename in line 77 of `app.py` matches your intended checkpoint.

## Suggestions for new models
The most promising models right now are 15_remap_infix and 26_highdrop_remap. 26 is currently running on the server. I recommend using those as starting points for future iteration. 
Some of my ideas for improving the models:
- 15 uses dropout = 0. 26 has dropout = 0.3. Try increasing dropout a bit more.
- Learning rate could be adjusted either up or down. It also helps to "bump" the LR if the model seems to get stuck during training. 
To do this, stop training, increase the LR parameter, and rerun `sbatch run_xx.job`. This will resume training from the last checkpoint, but with a higher LR.
- Experiment with number of warmup steps. Warmup steps = 0 causes divide by 0, so you need to set it to at least 1.
- So far, the best model dimensions seem to be 2 encoder and 2 decoder layers, with model dimension 40 and feedforward dimension (fairseq calls this embedding dimension) 160.
If you feel particularly ambitious, you can experiment with 3-4 of each layer and smaller dimension. We've observed that the FF dimension should be 4*model dim., and performance on this task degrades when model dim >=128.
Details of the experiments I've conducted so far can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/16eKj7OuKgxJMHutO9Jm50soAb_mfYcv0Rn10UtdbA7A/edit?usp=sharing) - it's not the neatest, but I can answer questions if you have them.

---------

## ACKNOWLEDGEMENTS
* This app was developed and models were trained by Katy Felkner, USC ISI, felkner <at> isi <dot> edu. 

* The code for the web app is *heavily* based on RTG-serve from 
[Reader-Translator-Generator](https://github.com/isi-nlp/rtg/) by Thamme Gowda. 
RTG-serve documentation can be found [here](https://cutelab.name/rtg/#_rtg_serve). Thank you, TG!

* This work is support by the DARPA ReMATH program under contract number XXXXXXXX.
* TODO Acknowledgement text