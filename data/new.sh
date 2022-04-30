#!/bin/bash

mkdir -p ACE2005/tmp

FOLDER="../json"

python transfer.py transfer $FOLDER/train.json ACE2005/tmp/train.json [PER-SOC]
python transfer.py transfer $FOLDER/dev.json ACE2005/tmp/dev.json [PER-SOC]
python transfer.py transfer $FOLDER/test.json ACE2005/tmp/test.json [PER-SOC]

python new_process.py process ACE2005/tmp/train.json ACE2005/ent_rel_file.json ACE2005/train.json bert-base-uncased
python new_process.py process ACE2005/tmp/dev.json ACE2005/ent_rel_file.json ACE2005/dev.json bert-base-uncased
python new_process.py process ACE2005/tmp/test.json ACE2005/ent_rel_file.json ACE2005/test.json bert-base-uncased

rm -rf ACE2005/tmp
