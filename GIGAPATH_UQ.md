# env 
conda env create -f autoi-env.yml
conda activate autoi

# slideflow
cd ~/slideflow
git checkout gigapath
pip install -e .

# dbx
cd ~/dbx
pip install -e .

# autoi
cd ~/autoi
pip install -e .

# dependencies (in addition to or overriding those in the env, slideflow and structured_eye)
# TODO: incorporate into the appropriate requirements file
pip install timm==1.0.15
pip install xformers==0.0.20
pip install dinov2 --extra-index-url https://pypi.nvidia.com # import dinov2 fails
pip install omegaconf==2.3.0
pip install fvcore==0.1.5.post20221221

