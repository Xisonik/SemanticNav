FROM nvcr.io/nvidia/isaac-sim:4.5.0
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
      apt-get -y install sudo python3 python3-pip curl git
RUN apt-get update && apt-get install -y --allow-downgrades libbrotli1=1.0.9-2build6
RUN apt install -y build-essential cmake git libssl-dev libfreetype6-dev zlib1g-dev \
	libbz2-dev libreadline-dev libsqlite3-dev libglib2.0-0 libfontconfig1-dev curl git \
	libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev ncurses-term \
	libffi-dev liblzma-dev libosmesa6-dev patchelf wget unzip
RUN mkdir -p /root/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh
RUN bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3
RUN rm -f /root/miniconda3/miniconda.sh

RUN mkdir /root/isaacsim
WORKDIR /root/isaacsim
RUN wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip
RUN unzip isaac-sim-standalone-4.5.0-linux-x86_64.zip
RUN rm -f isaac-sim-standalone-4.5.0-linux-x86_64.zip

RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN mkdir /root/SemanticNav
WORKDIR /root/SemanticNav
RUN git init
RUN git pull git@github.com:Xisonik/SemanticNav.git main:master

RUN pip install gdown
RUN mkdir /root/tmp
WORKDIR /root/tmp
RUN gdown --folder https://drive.google.com/drive/folders/1sdWFHsREqW_2fmu2E2mjV8LxF6KUYaSe?usp=sharing -O /root/tmp
RUN rm -rf /root/SemanticNav/source/isaaclab_assets/data/
RUN mkdir -p /root/SemanticNav/source/isaaclab_assets/data/
RUN unzip aloha_assets.zip -d /root/SemanticNav/source/isaaclab_assets/data/
RUN rm -rf /root/SemanticNav/data/all_paths.json
RUN mkdir /root/SemanticNav/data/
RUN mv all_paths.json /root/SemanticNav/data/
RUN rm -rf /root/SemanticNav/source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/text_embeddings.pt
RUN mv text_embeddings.pt /root/SemanticNav/source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/
RUN cd /root
RUN rm -rf /root/tmp

ENV ISAACSIM_PATH="/root/isaacsim"
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
WORKDIR /root/SemanticNav
RUN ln -s ${ISAACSIM_PATH} _isaac_sim
RUN (source /root/miniconda3/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main)
RUN (source /root/miniconda3/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r)
RUN (source /root/miniconda3/etc/profile.d/conda.sh && conda run ./isaaclab.sh --conda .semantic_nav)
RUN (source /root/miniconda3/etc/profile.d/conda.sh && conda activate .semantic_nav && ./isaaclab.sh --install)
RUN echo 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate .semantic_nav' >> /root/.bashrc
RUN source /root/.bashrc
CMD ["bash"]
