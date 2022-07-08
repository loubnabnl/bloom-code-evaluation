import json
import multiprocessing
import os

from datasets import  load_metric
from transformers import HfArgumentParser

from arguments import HumanEvalArguments

def merge_generations(output_folder, num_tasks):
    """merge the generations and references files of the tasks into single lists of all tasks"""
    generations_exp1 = []
    references_exp1 = []

    generations_exp2 = []
    references_exp2 = []

    generations_exp3 = []
    references_exp3 = []

    #fill each list with the generations and references of each experiment
    for task in range(1, num_tasks + 1):
        task_path_exp1 = f"bloom/task_{task}_{task+1}/output_1"
        task_path_exp2 = f"bloom/task_{task}_{task+1}/output_2"
        task_path_exp3 = f"bloom/task_{task}_{task+1}/output_3"

        with open(f"{task_path_exp1}/generations.json") as f:
            # list of a list with 200 generations
            gens = json.load(f)
            generations_exp1.append(gens[0])
        with open(f"{task_path_exp1}/references.json") as f:
            # list with one test case inside
            refs = json.load(f)
            references_exp1.append(refs[0])

        with open(f"{task_path_exp2}/generations.json") as f:
            gens = json.load(f)
            generations_exp2.append(gens[0])
        with open(f"{task_path_exp2}/references.json") as f:
            refs = json.load(f)
            references_exp2.append(refs[0])
                
        with open(f"{task_path_exp3}/generations.json") as f:
            gens = json.load(f)
            generations_exp3.append(gens[0])
        with open(f"{task_path_exp3}/references.json") as f:
            refs = json.load(f)
            references_exp3.append(refs[0])

    return generations_exp1, references_exp1, generations_exp2, references_exp2, generations_exp3, references_exp3

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
    generations_exp1, references_exp1, generations_exp2, references_exp2, generations_exp3, references_exp3 = merge_generations(output_folder=args.output_file, num_tasks=args.num_tasks)

    # Experiment 1
    print("Evaluating completions for experiment 1 (temperature 0.2)")
    pass_at_k, _ = code_eval_metric.compute(
        references=references_exp1, predictions=generations_exp1, num_workers=args.num_workers
    )
    print(f"Results temperature 0.2: {pass_at_k}")

    with open(f"{args.output_file}/eval_results_exp1.json", "w") as fp:
        json.dump(pass_at_k, fp)

    # Experiment 2
    print("Evaluating completions for experiment 2 (temperature 0.6)")
    pass_at_k, _ = code_eval_metric.compute(
        references=references_exp2, predictions=generations_exp2, num_workers=args.num_workers
    )
    print(f"Results temperature 0.6: {pass_at_k}")

    with open(f"{args.output_file}/eval_results_exp2.json", "w") as fp:
        json.dump(pass_at_k, fp)

    # Experiment 3
    print("Evaluating completions for experiment 3 (temperature 0.8)")
    pass_at_k, _ = code_eval_metric.compute(
        references=references_exp3, predictions=generations_exp3, num_workers=args.num_workers
    )
    print(f"Results temperature 0.8: {pass_at_k}")

    with open(f"{args.output_file}/eval_results_exp3.json", "w") as fp:
        json.dump(pass_at_k, fp)


# For some reason the folliwng seems to be necessary sometimes for code_eval to work nice with multiprocessing
# https://stackoverflow.com/questions/60804599/python-multiprocessing-keeps-spawning-the-whole-script
if __name__ == "__main__":
    main()
