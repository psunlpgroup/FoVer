def get_error_labels_stats(verification_results: list[dict]) -> dict:
    num_instances = len(verification_results)
    
    if num_instances == 0:
        return {}
    
    # y_correct does not always exist
    if "y_correct" in verification_results[0].keys():
        y_correct_list = [d["y_correct"] for d in verification_results]
        if None in y_correct_list:
            y_correct_count = None
        else:
            y_correct_count = sum(y_correct_list)
    else:
        y_correct_count = None
    
    all_step_correct_count = sum(
        [d["all_process_correct"] for d in verification_results]
    )
    
    # step level
    all_steps_level_label = [step for d in verification_results
                             for step in d["proof_step_correctness"]]
    num_steps = len(all_steps_level_label)
    num_correct_steps = sum(all_steps_level_label)
    
    return {
        "num_instances": num_instances,
        "y_correct": y_correct_count,
        "%y_correct": y_correct_count / num_instances * 100 \
            if y_correct_count is not None else None,
        "all_step_correct": all_step_correct_count,
        "%all_step_correct": all_step_correct_count / num_instances * 100,
        "num_steps": num_steps,
        "num_correct_steps": num_correct_steps,
        "%correct_steps": num_correct_steps / num_steps * 100,
    }
