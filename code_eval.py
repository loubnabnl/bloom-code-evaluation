import json
import multiprocessing
import os
import re

from datasets import load_dataset
from tqdm import tqdm

import torch
import transformers
from arguments import HumanEvalArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)


EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("|".join(EOF_STRINGS), string)[0].rstrip()


def complete_code(model, tokenizer, prompt, num_completions=1, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    prompt = tokenizer.eos_token + prompt

    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    tokenized_prompt = {k: v.to("cuda") for k, v in tokenized_prompt.items()}

    outputs = model.generate(**tokenized_prompt, num_return_sequences=num_completions, **gen_kwargs)
    code_gens = [tokenizer.decode(outputs[i]) for i in range(outputs.shape[0])]

    return [first_block(code_gen[len(prompt) :]) for code_gen in code_gens]

def get_args():
    parser = HfArgumentParser(HumanEvalArguments)
    parser.add_argument("--task_start", type=int, required=True)
    parser.add_argument("--task_end", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--max_memory_per_gpu", type=str, default="50GB")
    return parser.parse_args()

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

def main():
    # Setup configuration
    args = get_args()

    transformers.logging.set_verbosity_error()
    # make sure tokenizer plays nice with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()

    set_seed(args.seed)

    print(f"loading tokenizer and model in {args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_ckpt, 
        device_map="auto", 
        torch_dtype=getattr(torch, args.dtype),
        max_memory=get_gpus_max_memory(args.max_memory_per_gpu),
        offload_folder="offload",
    )
    print("model loaded")

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stopping_criteria": StoppingCriteriaList([EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]),
    }

    # Load evaluation dataset and metric
    human_eval = load_dataset("openai_humaneval")

    # Generate completions for evaluation set
    print("Starting code generation")
    generations, references = [], []
    for task in tqdm(range(args.task_start, args.task_end)):
        task_generations = []
        prompt = human_eval["test"][task]["prompt"].strip()
        gen_kwargs["stopping_criteria"][0].start_length = len(tokenizer(prompt)["input_ids"])
        for batch in tqdm(range(args.n_samples // args.batch_size)):
            task_generations.extend(complete_code(model, tokenizer, prompt, num_completions=args.batch_size, **gen_kwargs))
        generations.append([prompt + gen for gen in task_generations])
        test_func = human_eval["test"][task]["test"]
        entry_point = f"check({human_eval['test'][task]['entry_point']})"
        references.append("\n" + test_func + "\n" + entry_point)
        
    # Save results to json file
    os.makedirs(args.output_file, exist_ok=True)
    reference_file = f"{args.output_file}/references.json"
    generation_file = f"{args.output_file}/generations.json"

    with open(reference_file, "w") as fp:
        json.dump(references, fp)
    with open(generation_file, "w") as fp:
        json.dump(generations, fp)


if __name__ == "__main__":
    main()
