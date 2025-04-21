import sys

sys.path.append("./")


def evaluate_single_prompt(evaluator, args, prompt_pre, output=None):
    return evaluator.forward(prompt_pre)


def evaluate_optimized_prompt(population, pop_marks, out_path, evaluator, args):
    with open(
        out_path,
        "w",
    ) as wf:
        prompts, marks, all_scores, scores_strs = [], [], [], []

        for prompt, mark in zip(population, pop_marks):
            scores = evaluator.forward(prompt, test=True)
            all_scores.append(scores[-1])
            scores_str = "\t".join([str(round(s, 4)) for s in scores])
            wf.write(f"{mark}\t{prompt}\t{scores_str}\n")
            scores_strs.append(scores_str)
            marks.append(mark)
            prompts.append(prompt)
            wf.flush()
        score_sorted, prompts_sorted, mark_sorted, scores_strs_sorted = (
            list(t)
            for t in zip(
                *sorted(zip(all_scores, prompts, marks, scores_strs), reverse=True)
            )
        )

        wf.write("\n----------sorted results----------\n")
        for i in range(len(score_sorted)):
            wf.write(
                f"{mark_sorted[i]}\t{prompts_sorted[i]}\t{scores_strs_sorted[i]}\n"
            )
        wf.close()
    return score_sorted[0], prompts_sorted[0]

