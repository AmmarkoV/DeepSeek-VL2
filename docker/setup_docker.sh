#!/usr/bin/env bash
# This script builds and runs a docker image for local use.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..
REPOSITORY=`pwd`

cd "$DIR"

#https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update



export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo nvidia-ctk runtime configure --runtime=docker


sudo apt install docker.io docker-buildx docker-compose-v2 docker-clean 

sudo systemctl restart docker

#Rootless mode
#nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
#systemctl --user restart docker
#sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place

sudo docker run hello-world


#Make sure docker group is ok
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker 

exit 0

