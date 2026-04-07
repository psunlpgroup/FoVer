import re


def split_steps(response: str) -> list[str]:
    """Split the response into steps using either a space or a newline.

    This treats one or more spaces and/or newlines as delimiters and returns
    the non-empty tokens in order.
    """
    if response is None:
        return []
    # Split on spaces or newlines (one or more), trim leading/trailing whitespace
    tokens = re.split(r"[ \n]+", response.strip())
    return [t for t in tokens if t != ""]


def get_solution_steps_from_response(response: str) -> list[str]:
    """ Get the solution steps from the given response.
    We assume that steps are separated by newlines in the response
    as instructed in the few-shot examples.
    
    Args:
        response (str): The response.
    
    Returns:
        solution_steps (list[str]): The solution steps.
    """
    
    # remove double newlines
    while "\n\n" in response:
        response = response.replace("\n\n", "\n")
    
    steps = response.split("\n")
    
    # remove empty steps
    solution_steps = [step for step in steps if len(step) > 0]
    
    # Merge short steps with the following step(s). If a step has fewer words
    # than `threshold_word_count`, we repeatedly concatenate the next step(s)
    # until the merged text meets the threshold. Special-case the final step:
    # if the final step is very short or contains LaTeX closing markers (for
    # example `\]` or `\end{`), attach it to the previous step instead of
    # leaving it as a standalone step.
    threshold_word_count = 5
    merge_next_patterns = ["]", "\end{"]
    
    merged_steps = []
    i = 0
    while i < len(solution_steps):
        step = solution_steps[i]
        step_words = split_steps(step)
        
        # Final step is always kept
        if i == len(solution_steps) - 1:
            merged_steps.append(step)
            i += 1
            continue
        
        # If the step is long enough, keep it
        if len(step_words) >= threshold_word_count:
            merged_steps.append(step)
            i += 1
            continue
        
        ###
        # below this line: handle short steps
        
        # Special handling for the final step: merge it into the previous step
        # when a previous step exists and the final step contains an end marker.
        # Otherwise keep it as its own step.
        if i == len(solution_steps) - 1:
            merge_with_previous = True
            
            if len(merged_steps) == 0:
                merge_with_previous = False
            
            # If last step contains a closing LaTeX marker, attach to previous
            if not any(p in step for p in merge_next_patterns):
                merge_with_previous = False
            
            if merge_with_previous:
                merged_steps[-1] = merged_steps[-1] + "\n" + step
            else:
                merged_steps.append(step)
            
            i += 1
            continue
        
        # merge to previous step if the step contains a end marker
        if any(p in step for p in merge_next_patterns):
            if len(merged_steps) > 0:
                merged_steps[-1] = merged_steps[-1] + "\n" + step
                i += 1
                continue

        # Merge short steps with following steps until length threshold met
        # However, the final step is not merged here because it is always kept as is.
        merged = step
        j = i + 1
        while j < len(solution_steps) - 1 and len(split_steps(merged)) < threshold_word_count:
            merged = merged + "\n" + solution_steps[j]
            j += 1

        merged_steps.append(merged)
        i = j
    
    return merged_steps


if __name__ == "__main__":
    # test code for split_steps
    test_response = "Step one is reasonably long.\nStep two:\nStep three with more details.\n\].\n\\begin{align}\nStep four is also reasonably long.\nStep five is also reasonably long.\n\\end{align}\nFinal step."
    steps = split_steps(test_response)
    
    print("split_steps output:")
    for idx, step in enumerate(steps):
        print(f"Step {idx + 1}: {step}")
        print(f"  (word count: {len(split_steps(step))})")
    
    print("\nget_solution_steps_from_response output:")
    steps = get_solution_steps_from_response(test_response)
    for idx, step in enumerate(steps):
        print(f"Step {idx + 1}: {step}")
        print(f"  (word count: {len(split_steps(step))})")
