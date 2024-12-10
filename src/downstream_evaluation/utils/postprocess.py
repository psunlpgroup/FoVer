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
    
    return solution_steps
