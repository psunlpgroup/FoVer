source $CONDA_SH

export PYTHONPATH="${PYTHONPATH}:./:../FoVer:../FLD-generator:../dtv:../skywork-o1-prm-inference"
export PATH=$PATH:$PWD/Isabelle2022/bin
export PISA_PATH=$PWD/Portal-to-ISAbelle/src/main/python


###
# LLM Inference
bash src/llm/setup.sh

###
# LLM training
bash setup/setup_llama_factory.sh

###
# SkyWork PRM evaluation
git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git ../skywork-o1-prm-inference

conda create -n skywork_prm python=3.9
pip install vllm==v0.6.4.post1
cd skywork-o1-prm-inference
pip install -e .
cd ../FoVer

conda deactivate

###
# Dataset creation

# conad environment for dataset creation
conda env create -f setup/environment_data_creation.yml
conda activate fover_data_creation

###
# First-order logic dataset (FLDx2)

## run the following command
# ipython
# > import nltk
# > nltk.download('cmudict')

# FLD
git clone git@github.com:hitachi-nlp/FLD-generator.git ../FLD-generator
cd ../FLD-generator
git checkout 00d12c4a9132a4fb43cd77f24db03ea7f5b27877
cd ../FoVer

# import requirements
pip install -r ../FLD-generator/requirements/requrements.txt

conda deactivate

###
# Math theorem proof (Isabelle)

# Isabelle 2022
wget https://isabelle.in.tum.de/website-Isabelle2022/dist/Isabelle2022_linux.tar.gz
tar -xzf Isabelle2022_linux.tar.gz
rm Isabelle2022_linux.tar.gz

# PISA  https://github.com/albertqjiang/Portal-to-ISAbelle
# also refer to https://github.com/wellecks/ntptutorial/blob/80f56388c1e22004c8e63f6faa5c8d3b23b2e650/partII_dsp/isabelle_setup.md
git clone https://github.com/albertqjiang/Portal-to-ISAbelle.git
git checkout 56def2c

pip install -r ./Portal-to-ISAbelle/requirements.txt

# install PISA dependencies -- sdkman
conda install conda-forge::zip
export SDKMAN_DIR="$HOME/.sdkman"  # you can change this to your preferred location
curl -s "https://get.sdkman.io" | bash
[[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]] && source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk help

sdk install java 11.0.11-open
sdk install sbt

cd Portal-to-ISAbelle
sbt compile

# Build Isabelle HOL
cd ../
isabelle build -b -D Isabelle2022/src/HOL/ -j 20

wget https://archive.org/download/isabelle2022_afp20221206_heaps/isabelle2022heaps.tar.gz
tar -xzf isabelle2022heaps.tar.gz -C ./Isabelle2022-heaps
rm isabelle2022heaps.tar.gz
mv -n ./Isabelle2022-heaps/heaps/polyml-5.8.2_x86_64_32-linux/* ~/.isabelle/Isabelle2022/heaps/polyml-5.8.2_x86_64_32-linux/
rm -r ./Isabelle2022-heaps

# https://github.com/wellecks/ntptutorial/blob/80f56388c1e22004c8e63f6faa5c8d3b23b2e650/partII_dsp/isabelle_setup.md
cp ./src/dataset_creation/base_dataset_specific/isabelle/ntptutorial/Interactive.thy ./Isabelle2022/src/HOL/Examples/Interactive.thy

# https://github.com/albertqjiang/Portal-to-ISAbelle?tab=readme-ov-file#evaluation-setup-if-you-want-to-have-n-n50-pisa-servers-running-on-your-machine
# for parallel processing
BASE_DIR=$PWD
cd Portal-to-ISAbelle
sbt assembly
python eval_setup/copy_pisa_jars.py --pisa-jar-path target/scala-2.13/PISA-assembly-0.1.jar --number-of-jars 8 --output-path ../pisa_copy
# this process will make 8 * 35GB of files
python eval_setup/copy_isabelle.py --isabelle ../Isabelle2022 --isabelle-user ${HOME}/.isabelle --number-of-copies 8 --output-path ${BASE_DIR}/isabelle_copy
