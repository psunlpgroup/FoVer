from typing import Literal, TypedDict

FOL_PROOF_LABEL = Literal["PROVED", "DISPROVED", "UNKNOWN"]


class FolStep(TypedDict):
    """ Represents a step in a first-order logic proof: assumptions -> conclusion.
    "assumptions" and "conclusions" are logical expressions in string format, such as "(¬{B} & ¬{C})"
    
    Attributes:
        assumptions (str): A logical expression representing the assumptions.
        conclusion (str): A logical expression representing the conclusion.
    """
    
    assumptions: str
    conclusion: str


VERIFICATION_LABEL = Literal["correct", "wrong_implication"]
