from FLD_generator.formula import Formula, negate
from FLD_generator.formula_checkers.z3_logic_checkers.checkers import check_sat


def is_logically_equal(fact1: Formula, fact2: Formula) -> bool:
    """ Checks if two logical expressions are logically equivalent.
    
    Args:
        fact1 (Formula): The first logical expression.
        fact2 (Formula): The second logical expression.
    
    Returns:
        result (bool): True if the two expressions are logically equivalent, False otherwise.
    """
    
    # fact1 -> fact2
    sat1 = check_sat([fact1, negate(fact2)])
    
    # fact2 -> fact1
    sat2 = check_sat([fact2, negate(fact1)])
    
    return (not sat1) and (not sat2)
