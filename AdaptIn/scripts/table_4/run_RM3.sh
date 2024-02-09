python ../../Non_Seq_Rec_Model/AdaptIn_wo_LN.py --arch-sparse-feature-size=16 \
					--arch-mlp-bot="1-512-256-64-16" \
					--arch-mlp-top="512-256-1" \
					--arch-interaction-op=transformers \
					--num-encoder-layers=1 \
					--num-attention-heads=2 \
					--feedforward-dim=128 \
					--dropout=0.01 \
					--norm-first=False \
					--activation=relu \
					--mask-threshold=0.001-0.01 \
					--data-generation=dataset \
					--data-set=avazu \
					--avazu-db-path=<path_to_avazu_db_file> \
					--avazu-train-file=<path_to_avazu_train> \
					--avazu-test-file=<path_to_avazu_test> \
					--loss-function=<loss_fucntion> \
					--round-targets=True \
					--learning-rate=0.2 \
					--mini-batch-size=128 \
					--print-freq=4096 \
					--print-time \
					--test-mini-batch-size=16384 \
					--test-num-workers=12 \
					--test-freq=4096 \
					--nepochs=1 \
					--mlperf-logging \
					--numpy-rand-seed=123