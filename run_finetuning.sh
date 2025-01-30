export CUDA_VISIBLE_DEVICES=1,2
accelerate launch instruction_finetuning.py --model_id mistralai/Mistral-7B-v0.3\
					--last_checkpoint checkpoint-60

