#!/bin/bash

## problem is that 
## File "eval_adipose_front.py", line 73, in main
##  device,
## File "eval_adipose_front.py", line 96, in segment_eval_pipeline
## eval_adipose.run_segment_eval(

## also that I overwrote some of files for example the one where the predictions are stored eval_runs.py or something like that...?

python eval_adipose_front.py --project-name test --organ-name adipose --seg-model-id 1 --slide-id 1 
