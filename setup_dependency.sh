#! /bin/bash
#
# setup_custom.sh
# Copyright (C) 2020-06-04 Junxian <He>
#
# Distributed under terms of the MIT license.
#


pip install --editable .
pip install tensorboardX
pip install nlp
pip install stanza
pip install spacy
pip install edlib
pip install sklearn

# install files2rouge
pip install -U git+https://github.com/pltrdy/pyrouge

git clone https://github.com/pltrdy/files2rouge.git
cd files2rouge
python setup_rouge.py
python setup.py install

sudo apt update && apt-get upgrade
sudo apt install openssh-server openssh-client
apt-get install default-jdk

