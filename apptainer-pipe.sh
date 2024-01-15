#!/bin/bash
set -e
# This script is for inference


VERSION=1.2.10
# Stop on error, if set to 1 will exit program if any of the docker commands fail
set -e
STOPONERROR=1

# enable the baseline detection, 0 for disablle
BASELINELAYPA=1


LAYPAMODEL=/cephyr/users/zhihaoy/Alvis/Downloads/config.yaml
LAYPAMODELWEIGHTS=/cephyr/users/zhihaoy/Alvis/Downloads/model_best_mIoU.pth

# enable the HTR transcribing, 0 for disable (of course we want!)
HTRLOGHI=1
HTRLOGHIMODEL=/cephyr/users/zhihaoy/Alvis/loghi/best_val_nov30

# set this to 1 for recalculating reading order, line clustering and cleaning.
RECALCULATEREADINGORDER=1
# if the edge of baseline is closer than x pixels...
RECALCULATEREADINGORDERBORDERMARGIN=50
# clean if 1
RECALCULATEREADINGORDERCLEANBORDERS=0
# how many threads to use
RECALCULATEREADINGORDERTHREADS=4

#detect language of pagexml, set to 1 to enable, disable otherwise
DETECTLANGUAGE=0
#interpolate word locations
SPLITWORDS=1
#BEAMWIDTH: higher makes results slightly better at the expense of lot of computation time. In general don't set higher than 10
BEAMWIDTH=2
#used gpu ids, set to "-1" to use CPU, "0" for first, "1" for second, etc
GPU=0

USE2013NAMESPACE=" -use_2013_namespace "

# Something real
if [ -z $1 ]; then echo "Please provide the path to images to be HTR-ed" && exit 1; fi;
tmpdir=$(mktemp -d)
echo $tmpdir

DOCKERGPUPARAMS=""
if [[ $GPU -gt -1 ]]; then
        DOCKERGPUPARAMS="--nv"
        echo "using GPU ${GPU}"
fi

SRC=$1

mkdir $tmpdir/imagesnippets/
mkdir $tmpdir/linedetection
mkdir $tmpdir/output


find $SRC -name '*.done' -exec rm -f "{}" \;


if [[ $BASELINELAYPA -eq 1 ]]
then
        echo "Starting Laypa baseline detection"
        # inference baseline of input images with Laypa
        input_dir=$SRC
        output_dir=$SRC
        LAYPADIR="$(dirname "${LAYPAMODEL}")"

        if [[ ! -d $input_dir ]]; then
                echo "Specified input dir (${input_dir}) does not exist, stopping program"
                exit 1
        fi

        if [[ ! -d $output_dir ]]; then
                echo "Could not find output dir (${output_dir}), creating one at specified location"
                mkdir -p $output_dir
        fi
        apptainer exec --nv laypa.sif \
        python laypa-c46490c8fbdb78795bddd9c192b8958d941b5e27/run.py \
        -c $LAYPAMODEL \
        -i $input_dir \
        -o $output_dir \
        --opts MODEL.WEIGHTS "" TEST.WEIGHTS $LAYPAMODELWEIGHTS | tee -a $tmpdir/log.txt

        # > /dev/null

        if [[ $STOPONERROR && $? -ne 0 ]]; then
                echo "Laypa errored has errored, stopping program"
                exit 1
        fi

        apptainer exec tool.sif /src/loghi-tooling/minions/target/appassembler/bin/MinionExtractBaselines \
        -input_path_png $output_dir/page/ \
        -input_path_page $output_dir/page/ \
        -output_path_page $output_dir/page/ \
        -as_single_region true \
        -laypaconfig $LAYPAMODEL $USE2013NAMESPACE | tee -a $tmpdir/log.txt


        if [[ $STOPONERROR && $? -ne 0 ]]; then
                echo "MinionExtractBaselines (Laypa) errored has errored, stopping program"
                exit 1
        fi
fi

# #HTR option 1 LoghiHTR
if [[ $HTRLOGHI -eq 1 ]]
then
        # cut the big image into small pieces to transcribe
        echo "starting Loghi HTR"
       apptainer exec tool.sif /src/loghi-tooling/minions/target/appassembler/bin/MinionCutFromImageBasedOnPageXMLNew \
       -input_path $SRC \
       -outputbase $tmpdir/imagesnippets/ \
       -output_type png \
       -channels 4 \
       -threads 4 $USE2013NAMESPACE| tee -a $tmpdir/log.txt


        if [[ $STOPONERROR && $? -ne 0 ]]; then
                echo "MinionCutFromImageBasedOnPageXMLNew has errored, stopping program"
                exit 1
        fi

       find $tmpdir/imagesnippets/ -type f -name '*.png' > $tmpdir/lines.txt

	LOGHIDIR="$(dirname "${HTRLOGHIMODEL}")"
        # running loghi htr model and inference small image pieces
	bash -c "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python3 /src/loghi-htr/src/main.py \
        --do_inference \
        --existing_model $HTRLOGHIMODEL  \
        --batch_size 64 \
        --use_mask \
        --inference_list $tmpdir/lines.txt \
        --results_file $tmpdir/results.txt \
        --charlist $HTRLOGHIMODEL/charlist.txt \
        --gpu $GPU \
        --output $tmpdir/output/ \
        --config_file_output $tmpdir/output/config.json \
        --beam_width $BEAMWIDTH " | tee -a $tmpdir/log.txt

        if [[ $STOPONERROR && $? -ne 0 ]]; then
                echo "Loghi-HTR has errored, stopping program"
                exit 1
        fi
        apptainer run tool.sif /src/loghi-tooling/minions/target/appassembler/bin/MinionLoghiHTRMergePageXML \
                -input_path $SRC/page \
                -results_file $tmpdir/results.txt \
                -config_file $tmpdir/output/config.json $USE2013NAMESPACE | tee -a $tmpdir/log.txt


        if [[ $STOPONERROR && $? -ne 0 ]]; then
                echo "MinionLoghiHTRMergePageXML has errored, stopping program"
                exit 1
        fi
fi

if [[ $RECALCULATEREADINGORDER -eq 1 ]]
then
        echo "recalculating reading order"
        if [[ $RECALCULATEREADINGORDERCLEANBORDERS -eq 1 ]]
        then
                echo "and cleaning"
                apptainer run tool.sif /src/loghi-tooling/minions/target/appassembler/bin/MinionRecalculateReadingOrderNew \
                        -input_dir $SRC/page/ \
			-border_margin $RECALCULATEREADINGORDERBORDERMARGIN \
			-clean_borders \
			-threads $RECALCULATEREADINGORDERTHREADS $USE2013NAMESPACE | tee -a $tmpdir/log.txt

                if [[ $STOPONERROR && $? -ne 0 ]]; then
                        echo "MinionRecalculateReadingOrderNew has errored, stopping program"
                        exit 1
                fi
        else
                apptainer run tool.sif /src/loghi-tooling/minions/target/appassembler/bin/MinionRecalculateReadingOrderNew \
                        -input_dir $SRC/page/ \
			-border_margin $RECALCULATEREADINGORDERBORDERMARGIN \
			-threads $RECALCULATEREADINGORDERTHREADS $USE2013NAMESPACE| tee -a $tmpdir/log.txt

                if [[ $STOPONERROR && $? -ne 0 ]]; then
                        echo "MinionRecalculateReadingOrderNew has errored, stopping program"
                        exit 1
                fi
        fi
fi
if [[ $DETECTLANGUAGE -eq 1 ]]
then
        echo "detecting language..."
        apptainer run tool.sif /src/loghi-tooling/minions/target/appassembler/bin/MinionDetectLanguageOfPageXml \
                -page $SRC/page/ $USE2013NAMESPACE | tee -a $tmpdir/log.txt


        if [[ $STOPONERROR && $? -ne 0 ]]; then
                echo "MinionDetectLanguageOfPageXml has errored, stopping program"
                exit 1
        fi
fi


if [[ $SPLITWORDS -eq 1 ]]
then
        echo "MinionSplitPageXMLTextLineIntoWords..."
        apptainer run tool.sif /src/loghi-tooling/minions/target/appassembler/bin/MinionSplitPageXMLTextLineIntoWords \
                -input_path $SRC/page/ $USE2013NAMESPACE | tee -a $tmpdir/log.txt

        if [[ $STOPONERROR && $? -ne 0 ]]; then
                echo "MinionSplitPageXMLTextLineIntoWords has errored, stopping program"
                exit 1
        fi
fi

# cleanup results
rm -rf $tmpdir

