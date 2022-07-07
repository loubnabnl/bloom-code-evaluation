from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HumanEvalArguments:
    """
    Configuration for running evaluation on HumanEval dataset.
    """

    model_ckpt: Optional[str] = field(
        default="bgscience/bloom", metadata={"help": "Model name or path of model to be evaluated."}
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})
    num_tasks: Optional[int] = field(
        default=None,
        metadata={"help": "The number of human-eval tasks to run. If not included all tasks are evaluated."},
    )
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Sample from the language model's output distribution."}
    )
    temperature: Optional[float] = field(default=0.2, metadata={"help": "Sampling temperature used for generation."})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "Maximum number of newly generated tokens."})
    top_k: Optional[int] = field(default=0, metadata={"help": "Top-k parameter used for generation."})
    top_p: Optional[float] = field(default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."})
    batch_size: Optional[int] = field(default=1, metadata={"help": "Number of generations to run in parallel."})
    n_samples: Optional[int] = field(
        default=200, metadata={"help": "Number of completions to generate for each sample."}
    )
    seed: Optional[int] = field(default=1, metadata={"help": "Random seed used for evaluation."})
    output_file: Optional[str] = field(
        default="code_generations", metadata={"help": "folder to save code generations and references"}
    )
    device_int: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "Determine which device to run the `text-generation` Pipeline on. -1 is CPU and any zero or positive"
                " number corresponds to which GPU device id to run on."
            )
        },
    )
    HF_ALLOW_CODE_EVAL: Optional[str] = field(
        default="0", metadata={"help": "Allow `code_eval` to execute Python code on machine"}
    )