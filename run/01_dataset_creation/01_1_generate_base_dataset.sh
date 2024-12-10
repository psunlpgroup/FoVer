###
# base datasets for our dataset

# formal logic (fldx2)
# we remove cases where we cannot easily verify the step-level correctness
python src/dataset_creation/generate_base_dataset/preprocess_fldx2.py

# formal theorem proof (math for isabelle)
python src/dataset_creation/generate_base_dataset/preprocess_metamathqa.py --base_dataset gsm8k
python src/dataset_creation/generate_base_dataset/preprocess_big_math.py --domain math_word_problems
