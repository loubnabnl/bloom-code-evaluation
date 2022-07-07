export HF_DATASETS_OFFLINE=1

#setup model
OUTPUT_file=code_generations
MODEL_CKPT=bigscience/bloom
echo using $MODEL_CKPT as model checkpoint, if not done change it to a local repository

python  code_eval.py --model_ckpt $MODEL_CKPT \
--batch_size 1 \
--do_sample True \
--temperature 0.2 \
--top_p 0.95 \
--n_samples 200 \
--output_file $OUTPUT_file
