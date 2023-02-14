#!/usr/bin/env bash
#
# Script to setup an AWS EC2 instance with the following expected provisioning:
#
# - AMI: Amazon Linux 2 AMI with NVIDIA TESLA GPU Driver
# - Instance Type: p3.2xlarge (8 vCPU, 61 GiB, Tesla V100 GPU)
# - Storage: at least 32 GB
#

tgopt="$(cd "$(dirname "$0")"; cd ..; pwd)"

echo "export TERM=xterm-256color" >> ~/.bashrc

echo
echo ">> installing intel-tbb"
echo

sudo yum-config-manager --add-repo https://yum.repos.intel.com/tbb/setup/intel-tbb.repo
sudo rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo yum install -y intel-tbb
echo "source /opt/intel/tbb/bin/tbbvars.sh intel64" >> ~/.bashrc

echo
echo ">> installing conda"
echo

curl -sL -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
bash ~/miniconda.sh -b -p ~/.conda
rm ~/miniconda.sh
~/.conda/bin/conda init
source ~/.bashrc

echo
echo ">> installing python packages"
echo

pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r $tgopt/requirements.txt

echo
echo ">> compiling c++ extension"
echo

cd $tgopt/extension
python setup.py install

echo
echo ">> cleaning up"
echo

make clean
conda clean -a -y
pip cache purge
rm -rf ~/.cache

echo
echo ">> done! please restart your shell session"
