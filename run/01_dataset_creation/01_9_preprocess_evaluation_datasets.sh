###
# processbench
python src/dataset_creation/preprocess_direct_evaluation_datasets.py

###
# datasets for best-of-k

# MATH
wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar -P ../datasets
tar -xvf ../datasets/MATH.tar -C ../datasets
rm ../datasets/MATH.tar

# orca_math
python src/dataset_creation/generate_base_dataset/preprocess_big_math.py --dataset_name orca_math --domain math_word_problems

# hans
git clone git@github.com:tommccoy1/hans.git ../datasets/hans

# BBH
git clone git@github.com:suzgunmirac/BIG-Bench-Hard.git ../datasets/BIG-Bench-Hard
