def clean_up_isabelle_statement_and_proof(statement_or_proof: str) -> str:
    """ Remove comments from the proof. """
    lines = statement_or_proof.split("\n")
    cleaned_lines = []
    for line in lines:
        if len(line.strip()) == 0:
            continue
        
        if "(*" in line and "*)" in line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)
