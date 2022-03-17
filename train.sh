#!/bin/bash

run_with_adversaries() {
	log_dir="logs/train/with_adversaries/tradeoff_${1}/asr_${2}"
	save_dir="saves/with_adversaries/tradeoff_${1}/asr_${2}/lrs_${3}_${4}"
	mkdir -p $log_dir
	mkdir -p $save_dir
	python -u sailon_dann.py \
		--epochs 30 \
		--adversary-trade-off $1 \
		--lr ${3} \
		--adversary-lr ${4} \
		--save-dir $save_dir \
		--adversary-step-ratio ${2} \
		| tee "${log_dir}/lrs_${3}_${4}.txt"
}

run_without_adversaries() {
	log_dir="logs/train/without_adversaries"
	save_dir="saves/without_adversaries/lrs_${1}_${2}"
	mkdir -p $log_dir
	mkdir -p $save_dir
	python -u sailon_dann.py \
		--epochs 30 \
		--disable-adversaries \
		--lr ${1} \
		--adversary-lr ${2} \
		--save-dir $save_dir \
		| tee "${log_dir}/lrs_${1}_${2}.txt"
}

#run_without_adversaries 0.0005 0.0005

#run_with_adversaries 0.01 1.0 0.0005 0.0005
run_with_adversaries 0.1 1.0 0.0005 0.0005
#run_with_adversaries 0.5 1.0 0.0005 0.0005

#run_with_adversaries 0.01 2.0 0.0005 0.0005
run_with_adversaries 0.1 2.0 0.0005 0.0005
#run_with_adversaries 0.5 2.0 0.0005 0.0005

#run_with_adversaries 0.01 5.0 0.0005 0.0005
run_with_adversaries 0.1 5.0 0.0005 0.0005
#run_with_adversaries 0.5 5.0 0.0005 0.0005
