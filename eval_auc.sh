#!/bin/bash

run_with_adversaries() {
	log_dir="logs/eval_auc/with_adversaries/tradeoff_${1}/asr_${2}"
	mkdir -p $log_dir
	python -u eval_auc.py \
		--subject-classifier-load-file "saves/with_adversaries/tradeoff_${1}/asr_${2}/lrs_${3}_${4}/subject/best_accuracy.pth" \
		--verb-classifier-load-file "saves/with_adversaries/tradeoff_${1}/asr_${2}/lrs_${3}_${4}/verb/best_accuracy.pth" \
		--object-classifier-load-file "saves/with_adversaries/tradeoff_${1}/asr_${2}/lrs_${3}_${4}/object/best_accuracy.pth" \
		| tee "${log_dir}/lrs_${3}_${4}.txt"
}

run_without_adversaries() {
	log_dir="logs/eval_auc/without_adversaries"
	mkdir -p $log_dir
	python -u eval_auc.py \
		--subject-classifier-load-file "saves/without_adversaries/lrs_${1}_${2}/subject/best_accuracy.pth" \
		--verb-classifier-load-file "saves/without_adversaries/lrs_${1}_${2}/verb/best_accuracy.pth" \
		--object-classifier-load-file "saves/without_adversaries/lrs_${1}_${2}/object/best_accuracy.pth" \
		| tee "${log_dir}/lrs_${1}_${2}.txt"
}

run_without_adversaries 0.0005 0.0005

#run_with_adversaries 0.01 1.0 0.0005 0.0005
#run_with_adversaries 0.1 1.0 0.0005 0.0005
#run_with_adversaries 0.5 1.0 0.0005 0.0005

#run_with_adversaries 0.01 2.0 0.0005 0.0005
#run_with_adversaries 0.1 2.0 0.0005 0.0005
#run_with_adversaries 0.5 2.0 0.0005 0.0005

#run_with_adversaries 0.01 5.0 0.0005 0.0005
#run_with_adversaries 0.1 5.0 0.0005 0.0005
#run_with_adversaries 0.5 5.0 0.0005 0.0005
