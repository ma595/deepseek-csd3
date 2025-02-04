# Running DeepSeek-R1 on Cambridge CSD3

This provides some simple instructions on how to run DeepSeek-R1 on the nvidia A100 GPU nodes. 

TODO:
- [ ] Examine how to run mult-gpu and multi-node inference of [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B).

It is helpful to review the model's hardware requirements [here](https://dev.to/nodeshiftcloud/a-step-by-step-guide-to-install-deepseek-r1-locally-with-ollama-vllm-or-transformers-44a1). 

## Setting up environment
```
module purge
module load rhel8/default-amp
module load python/3.11.0-icl
```

## Installing Ollama

Two options, Spack or Mamba

### With Spack

This provides version 0.4.7

```
git clone --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git ./spack
source spack/share/spack/setup-env.sh
spack install ollama
# Takes some time to install.
```

### With Conda / Mamba

Install miniforge, then `mamba install ollama`

## Running Ollama

You need to pass ollama the environment variable `OLLAMA_MODELS`, otherwise it will fill up your HOME.
```
OLLAMA_MODELS=/path/to/rds/where/you/have/space ollama serve
# then open a new terminal and do:
ollama run deepseek-r1:14b`
```
This then enters a chat prompt - more complex models provide CoT. Obviously this can all be executed on a GPU if desired. 

```
srun -A <MY-ACCOUNT> -p ampere --gres=gpu:1 -n 1 -N 1 -t 01:00:0 --qos INTR --pty /bin/bash
```

## Running with Torch and Transformers.

It is also useful to symlink your $HOME/.config as this will become populated by huggingface (move pre-existing contents if necessary). 

```
ln -s /new/path/to/rds/.cache ~/.cache/
```
Download the DeepSeek-V3 repository. 
```
git clone https://github.com/deepseek-ai/DeepSeek-V3
python3 -m venv ./venv3
source venv3/bin/activate
cd DeepSeek-V3/inference; pip install -r "requirements.txt"
pip install huggingface
```

Then: 

`huggingface-cli download deepseek-ai/deepseek-r1-distill-llama-8b`


## Running Examples



