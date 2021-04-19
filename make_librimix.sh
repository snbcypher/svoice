#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Yossi Adi (adiyoss)

path=dataset/Libri2Mix
mix_type=both # choose 'both' for added noise or 'clean' for no noise
for x in $path $path/dev $path/test $path/train-100
do
    if [[ ! -e $x ]]; then
        mkdir -p $x
    fi
done
python -m svoice.data.audio ../librimix/data/Libri2Mix/wav8k/min/dev/mix_$mix_type > $path/dev/mix.json
python -m svoice.data.audio ../librimix/data/Libri2Mix/wav8k/min/dev/s1 > $path/dev/s1.json
python -m svoice.data.audio ../librimix/data/Libri2Mix/wav8k/min/dev/s2 > $path/dev/s2.json
python -m svoice.data.audio ../librimix/data/Libri2Mix/wav8k/min/test/mix_$mix_type > $path/test/mix.json
python -m svoice.data.audio ../librimix/data/Libri2Mix/wav8k/min/test/s1 > $path/test/s1.json
python -m svoice.data.audio ../librimix/data/Libri2Mix/wav8k/min/test/s2 > $path/test/s2.json
python -m svoice.data.audio ../librimix/data/Libri2Mix/wav8k/min/train-100/mix_$mix_type > $path/train-100/mix.json
python -m svoice.data.audio ../librimix/data/Libri2Mix/wav8k/min/train-100/s1 > $path/train-100/s1.json
python -m svoice.data.audio ../librimix/data/Libri2Mix/wav8k/min/train-100/s2 > $path/train-100/s2.json
