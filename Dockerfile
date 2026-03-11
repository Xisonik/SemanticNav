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

ENV ISAACSIM_PATH="/root/isaacsim"
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
WORKDIR /root/SemanticNav
RUN ln -s ${ISAACSIM_PATH} _isaac_sim
CMD ["bash"]