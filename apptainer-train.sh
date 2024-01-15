#!/bin/bash
set -e

# This shell script is for training process

VERSION=1.2.10

# Configuration for HTR mode selection
HTRLOGHI=1

# Model configuration
HTRLOGHIMODELHEIGHT=64
HTRBASEMODEL=/cephyr/users/zhihaoy/Alvis/loghi/best_val_nov30
#set to 1 to actually use basemodel, since we are using transfer learning, then 1.
USEBASEMODEL=1

# Define a VGSL model
HTRNEWMODEL="None,64,None,3 Cr3,3,24 Bn Mp2,2,2,2 Cr3,3,48 Bn Mp2,2,2,2 Cr3,3,96 Bn Cr3,3,96 Bn Mp2,2,2,2 Rc Bl256 Bl256 Bl256 Bl256 Bl256 O1s92"
# set channels to 1 to process input as grayscale, 3 for color, 4 for color and mask
channels=4

GPU=0

# Dataset and training configuration
listdir=/mimer/NOBACKUP/groups/snic2022-23-105/svenska_htr/trainingdata_all_LM
trainlist=$listdir/training_all_train.txt
validationlist=$listdir/training_all_val.txt
datadir=./scratch/republicprint
charlist=/cephyr/users/zhihaoy/Alvis/loghi/best_val_nov30/charlist.txt
epochs=12
height=$HTRLOGHIMODELHEIGHT
multiply=1

# 8 to 32 works fine
batch_size=32
model_name=myfirstmodel
learning_rate=0.0001

# Something real
tmpdir=$(mktemp -d)

mkdir -p $tmpdir/output

BASEMODEL=""
BASEMODELDIR=""
if [[ $USEBASEMODEL -eq 1 ]]; then
    BASEMODEL=" --existing_model "$HTRBASEMODEL
    BASEMODELDIR="-B $(dirname "${HTRBASEMODEL}"):$(dirname "${HTRBASEMODEL}")"

fi

# HTR option
if [[ $HTRLOGHI -eq 1 ]]; then
    echo "Starting Loghi HTR"
    # run HTR training process in the htr.sif
    apptainer exec --nv\
	$BASEMODELDIR \
        --bind $tmpdir:$tmpdir \
        --bind $listdir:$listdir \
        htr.sif python3 /src/loghi-htr/src/main.py \
        --do_train \
        --train_list $trainlist \
        --do_validate \
        --validation_list $validationlist \
        --learning_rate $learning_rate \
        --channels $channels \
        --batch_size $batch_size \
        --epochs $epochs \
        --gpu $GPU \
        --height $height \
        --use_mask \
        --seed 1 \
        --beam_width 1 \
        --model "$HTRNEWMODEL" \
        --multiply $multiply \
        --output $listdir \
        --model_name $model_name \
        --output_charlist $tmpdir/output_charlist.charlist \
        --output $tmpdir/output $BASEMODEL
fi

echo "Results can be found at:"
echo $tmpdir

