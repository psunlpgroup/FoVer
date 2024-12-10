conda env create -f setup/environment_llama_factory.yml
conda activate llama_factory_fover

# install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git ../LLaMA-Factory-FoVer
cd ../LLaMA-Factory-FoVer

git checkout 4a5d0f0
pip install -e ".[torch,metrics]"

cd ../FoVer

# additional dependencies
pip install deepspeed==0.16.4
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
