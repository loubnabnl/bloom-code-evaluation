import json
import multiprocessing
import os

from datasets import  load_metric
from tqdm import tqdm

from arguments import HumanEvalArguments
from transformers import HfArgumentParser

def main():
    # Setup configuration
    parser = HfArgumentParser(HumanEvalArguments)
    args = parser.parse_args()

    code_eval_metric = load_metric("code_eval")

    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = args.HF_ALLOW_CODE_EVAL

    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()

    # Run a quick test to see if code evaluation is enabled
    try:
        _ = code_eval_metric.compute(references=[""], predictions=[[""]])
    except ValueError as exception:
        print(
            'Code evaluation not enabled. Read the warning below carefully and then use `--HF_ALLOW_CODE_EVAL="1"` flag to enable code evaluation.'
        )
        raise exception

    # Load generations and references
    reference_file = f"{args.output_file}/references.json"
    generation_file = f"{args.output_file}/generations.json"
    with open(reference_file) as f:
        references = json.load(f)
    with open(generation_file) as f:
        generations = json.load(f)

    # Evaluate completions with "code_eval" metric
    pass_at_k, _ = code_eval_metric.compute(
        references=references, predictions=generations, num_workers=args.num_workers
    )
    print(f"Results: {pass_at_k}")

    # Save results to json file
    with open(f"{args.output_file}/eval_results.json", "w") as fp:
        json.dump(pass_at_k, fp)


# For some reason the folliwng seems to be necessary sometimes for code_eval to work nice with multiprocessing
# https://stackoverflow.com/questions/60804599/python-multiprocessing-keeps-spawning-the-whole-script
if __name__ == "__main__":
    main()
