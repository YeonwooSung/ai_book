from typing import List
from pyrouge import Rouge155
import os, shutil, random, string


def evaluate_rouge(
    summaries: List[List[str]],
    references: List[List[List[str]]],
    remove_temp: bool = False,
    rouge_args: List[str] = [],
):
    """
    Args:
        summaries: [[sentence]]. Each summary is a list of strings (sentences)
        references: [[[sentence]]]. Each reference is a list of candidate summaries.
        remove_temp: bool. Whether to remove the temporary files created during evaluation.
        rouge_args: [string]. A list of arguments to pass to the ROUGE CLI.
    """
    temp_dir = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    temp_dir = os.path.join("temp", temp_dir)
    print(temp_dir)
    system_dir = os.path.join(temp_dir, "system")
    model_dir = os.path.join(temp_dir, "model")
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)
    print(temp_dir, system_dir, model_dir)

    assert len(summaries) == len(references)
    for i, (summary, candidates) in enumerate(zip(summaries, references)):
        summary_fn = "%i.txt" % i
        for j, candidate in enumerate(candidates):
            candidate_fn = "%i.%i.txt" % (i, j)
            with open(os.path.join(model_dir, candidate_fn), "w") as f:
                # print(candidate)
                f.write("\n".join(candidate))

        with open(os.path.join(system_dir, summary_fn), "w") as f:
            f.write("\n".join(summary))

    args_str = " ".join(map(str, rouge_args))
    rouge = Rouge155(rouge_args=args_str)
    rouge.system_dir = system_dir  # type: ignore
    rouge.model_dir = model_dir  # type: ignore
    rouge.system_filename_pattern = r"(\d+).txt"
    rouge.model_filename_pattern = r"#ID#.\d+.txt"

    # rouge_args = '-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a'
    # output = rouge.convert_and_evaluate(rouge_args=rouge_args)
    output = rouge.convert_and_evaluate()

    r = rouge.output_to_dict(output)
    print(output)
    # print(r)

    # remove the created temporary files
    if remove_temp:
        shutil.rmtree(temp_dir)
    return r
