import os

from src.dataset_creation.base_dataset_specific.isabelle.ntptutorial.dsp_utils import Checker


def get_proof_checker(port: int = 9000, mod_id: int | None = None) -> Checker:
    """ Get the proof checker. """
    # get current working directory
    working_dir = os.getcwd()
    
    if mod_id is None:
        isa_path = f'{working_dir}/Isabelle2022'
    else:
        isa_path = f'{working_dir}/isabelle_copy/isabelle_copy_{mod_id}/main_isa/Isabelle2022'

    return Checker(
        working_dir=f'{isa_path}/src/HOL/Examples',
        isa_path=isa_path,
        theory_file=f'{isa_path}/src/HOL/Examples/Interactive.thy',
        port=port
    )


if __name__ == "__main__":
    theorem_and_sledgehammer_proof = """theorem example:
    assumes "gcd (n :: nat) 4 = 1" 
        and "lcm (n :: nat) 4 = 28"
    shows "n = 7"
proof -
    have h1: "1*28 = n*4"
        sledgehammer
    then have h2: "n = 1*28/4"
        sledgehammer
    thus ?thesis
        sledgehammer
qed"""

    mod_id = 4
    port = 8000 + mod_id
    checker = get_proof_checker(port=port, mod_id=mod_id)

    result = checker.check(theorem_and_sledgehammer_proof)
    print(result)

    print("\n==== Success: %s" % result['success'])
    print("--- Complete proof:\n%s" % result['theorem_and_proof'])
