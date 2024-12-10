""" Translate to natural language text (TODO) """

import json
from tqdm import tqdm
from pathlib import Path
import traceback
from copy import deepcopy

from src.config import splits_list
from src.path import get_error_labels_path

from src.dataset_creation.base_dataset_specific.fol.dataset_generation.\
    generate_error_labels import GenerateVerificationDatasetTap
from src.llm.utils import save_md5_hash
from src.dataset_creation.utils import get_error_labels_stats

print("Importing FLD_generator...")
from FLD_generator.formula import Formula
from FLD_generator.translators import build as build_translator
from FLD_generator.word_banks import build_wordbank


def get_symbols_from_problem(problem: str) -> tuple[str, list[tuple[str, str]]]:
    """ Extracts symbols from the problem text
    
    Args:
        problem (str): The problem text
    
    Returns:
        problem_format (str): The format of the problem text
            We will use this later to generate the natural language text
        problem_key_symbols (list[tuple[str, str]]): pairs of key and symbol
    """
    
    # problem = """$hypothesis$: ({A} & {B})
    # 
    # $context$:
    # fact1: \u00ac{M} -> ({F} & {I})
    # fact2: {T} -> \u00ac(\u00ac{N} & \u00ac{L})
    # ...
    
    problem_list = problem.split("\n")

    # get the hypothesis symbol
    hypothesis_symbol = problem_list[0].split(": ", 1)[1]
    key_symbols = [("hypothesis", hypothesis_symbol)]

    # get the facts symbols
    for line_num in range(3, len(problem_list)):
        key, symbol = problem_list[line_num].split(": ", 1)
        key_symbols.append((key, symbol))
    
    # we will use this later to generate the natural language text
    problem_format = """$hypothesis$: {hypothesis}

$context$:"""
    for fact_idx in range(1, len(key_symbols)):  # skip the hypothesis
        problem_format += f"\nfact{fact_idx}: {{fact{fact_idx}}}"
    
    return problem_format, key_symbols


def get_symbols_from_proof_steps(proof_steps: list[str]) \
        -> tuple[list[str], list[tuple[str, str]]]:
    """ Extracts symbols from the proof steps
    
    Args:
        proof_steps (list[str]): The proof steps
    
    Returns:
        proof_steps_format (list[str]): The format of the proof steps
            We will use this later to generate the natural language text
        proof_steps_key_symbols (list[tuple[str, str]]): pairs of key and symbol
    """
    
    # proof_steps = [
    #   "fact11 & fact7 -> int1: ({A} & {B})",
    #   "int1 -> hypothesis",
    #   "The final answer is PROVED",
    # ]
    
    proof_steps_format = []
    proof_steps_key_symbols = []
    
    for step in proof_steps:
        if ": " in step:
            prefix, symbol = step.split(": ", 1)
            key = prefix.split(" -> ", 1)[1]  # e.g., "int1"
            proof_steps_format.append(f"{prefix}: {{{key}}}")
            proof_steps_key_symbols.append((key, symbol))
        else:
            proof_steps_format.append(step)
    
    return proof_steps_format, proof_steps_key_symbols


def main():
    args = GenerateVerificationDatasetTap().parse_args()
    
    # https://github.com/hitachi-nlp/FLD-generator/blob/NeurIPS_2024/scripts/create_corpus.py
    translation_lang = "eng"
    translation_vocab = None
    
    print("Building word bank...")
    word_bank = build_wordbank(
        translation_lang,
        extra_vocab = translation_vocab if \
            translation_vocab not in [None, 'wordnet'] \
            else None
    )
    
    print("Building translator...")
    translator = build_translator(
        lang=translation_lang,
        config_name_or_path=["old-thing.v1"],
        word_bank=word_bank,
        adj_verb_noun_ratio='1-1-1',
        use_fixed_translation=True,
        reused_object_nouns_max_factor=1.0,
        limit_vocab_size_per_type=None,
        volume_to_weight="log10",
        default_weight_factor_type="W_VOL__1.0",
        knowledge_banks = [],
        no_transitive_object=True,
    )
    
    # translation
    for split in splits_list[::-1]:
        for seed in range(1, args.num_samples + 1):
            print(f"Split: {split}, Seed: {seed}")
            
            # load original error labels
            symbol_output_path = Path("../FoVer") / get_error_labels_path(
                dataset_name="fldx2_symbol", model_name=args.model_name,
                split=split, seed=seed
            )
            original_error_labels_path = \
                symbol_output_path.with_suffix(".full.jsonl")
            with open(original_error_labels_path, "r") as f:
                original_error_labels = [json.loads(line) for line in f]
            
            ###
            # Translate to natural language text
            
            # we will remove cases we get errors during translation
            # in those cases, we remove them from the symbol data as well
            error_labels_symbol = []
            error_labels_text = []
            
            for symbol_d in tqdm(original_error_labels):
                try:
                    # preprocess the problem
                    problem_format, problem_key_symbols = get_symbols_from_problem(
                        symbol_d["problem"]
                    )
                    
                    # preprocess the proof steps
                    proof_steps_format, proof_steps_key_symbols = \
                        get_symbols_from_proof_steps(symbol_d["proof_steps"]) 
                    
                    # prepare formulas
                    all_unique_formulas_str = \
                        [s[1] for s in problem_key_symbols] + \
                        [s[1] for s in proof_steps_key_symbols]
                    all_unique_formulas = [
                        Formula(f"({formula_str})") for formula_str in \
                            all_unique_formulas_str
                    ]

                    # translate
                    named_translations, _ = translator.translate(
                        all_unique_formulas, [], [], [], True
                    )
                except Exception as e:
                    print(f"Error for translating output on {symbol_d['id']} "
                          f"by {args.model_name}, skipping...")
                    full_traceback = traceback.format_exc()
                    print(full_traceback)
                    print()
                    print(all_unique_formulas)
                    continue
                
                # postprocess the translation
                translated_problem = \
                    named_translations[:len(problem_key_symbols)]
                translated_proof_steps = \
                    named_translations[len(problem_key_symbols):]
                
                # problem
                problem_key_text = {}
                for problem_idx in range(len(problem_key_symbols)):
                    key, _ = problem_key_symbols[problem_idx]
                    text = translated_problem[problem_idx][1]
                    problem_key_text[key] = text
                
                translated_problem = problem_format.format(**problem_key_text)
                
                # proof steps
                proof_steps_key_text = {}
                for proof_step_idx in range(len(proof_steps_key_symbols)):
                    key, _ = proof_steps_key_symbols[proof_step_idx]
                    text = translated_proof_steps[proof_step_idx][1]
                    proof_steps_key_text[key] = text
                
                translated_proof_steps = []
                for step in proof_steps_format:
                    for key, text in proof_steps_key_text.items():
                        step = step.replace(f"{{{key}}}", text)
                    
                    translated_proof_steps.append(step)
                
                # make the text data
                text_d = deepcopy(symbol_d)
                text_d["problem"] = translated_problem
                text_d["proof_steps"] = translated_proof_steps
                
                # save
                error_labels_symbol.append(symbol_d)
                error_labels_text.append(text_d)
            
            ###
            # save
            
            # symbol
            with open(symbol_output_path, "w") as f:
                for text_d in error_labels_symbol:
                    f.write(json.dumps(text_d) + "\n")
            save_md5_hash(symbol_output_path)
            
            symbol_stat = get_error_labels_stats(error_labels_symbol)
            with open(symbol_output_path.with_suffix(".stats.json"), "w") as f:
                json.dump(symbol_stat, f)
            
            # text
            text_output_path = Path("../FoVer") / get_error_labels_path(
                dataset_name="fldx2_text", model_name=args.model_name,
                split=split, seed=seed
            )
            text_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(text_output_path, "w") as f:
                for text_d in error_labels_text:
                    f.write(json.dumps(text_d) + "\n")
            save_md5_hash(text_output_path)
            
            text_stat = get_error_labels_stats(error_labels_text)
            with open(text_output_path.with_suffix(".stats.json"), "w") as f:
                json.dump(text_stat, f)


if __name__ == "__main__":
    main()
