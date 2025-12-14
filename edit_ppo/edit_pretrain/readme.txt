conda env create -f env.yaml

conda activate preview_flux

pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/huggingface/diffusers.git