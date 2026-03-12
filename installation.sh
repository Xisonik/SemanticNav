pip install gdown
mkdir /root/tmp
cd /root/tmp
gdown --folder https://drive.google.com/drive/folders/1sdWFHsREqW_2fmu2E2mjV8LxF6KUYaSe?usp=sharing -O /root/tmp
rm -rf /root/SemanticNav/source/isaaclab_assets/data/
mkdir -p /root/SemanticNav/source/isaaclab_assets/data/
unzip aloha_assets.zip -d /root/SemanticNav/source/isaaclab_assets/data/
rm -rf /root/SemanticNav/data/all_paths.json
mkdir /root/SemanticNav/data/
mv all_paths.json /root/SemanticNav/data/
rm -rf /root/SemanticNav/source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/text_embeddings.pt
mv text_embeddings.pt /root/SemanticNav/source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/
cd /root
rm -rf /root/tmp

cd /root/SemanticNav
export ISAACSIM_PATH="/root/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
ln -s ${ISAACSIM_PATH} ./_isaac_sim
source /root/miniconda3/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
source /root/miniconda3/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
source /root/miniconda3/etc/profile.d/conda.sh && conda run ./isaaclab.sh --conda .semantic_nav
source /root/miniconda3/etc/profile.d/conda.sh && conda activate .semantic_nav && ./isaaclab.sh --install
echo 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate .semantic_nav' >> /root/.bashrc