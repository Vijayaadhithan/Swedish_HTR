#!/bin/bash
VERSION=1.2.10
if [ -z $1 ]; then echo "please provide path to images and pagexml to be converted. The pageXML must be one level deeper than the images in a directory called \"page\"" && exit 1; fi;
if [ -z $2 ]; then echo "please provide output path" && exit 1; fi;
if [ -z $3 ]; then
        echo "setting numthreads=4"
        numthreads=4
else
        numthreads=$3
        echo "setting numthreads=$numthreads"
fi;

#directory containing images and pagexml. The pageXML must be one level deeper than the images in a directory called "page"
mkdir -p $2
inputdir=$(realpath $1/)
outputdir=$(realpath $2/)
filelist=$outputdir/training_all.txt
filelisttrain=$outputdir/training_all_train.txt
filelistval=$outputdir/training_all_val.txt
#training set split percentage, 90 for training, 10 for validating
trainsplit=90
DOCKERLOGHITOOLING=loghi/docker.loghi-tooling:$VERSION
INCLUDETEXTSTYLES=" -include_text_styles " # for further detect different text styles
SKIP_UNCLEAR=" -skip_unclear " # skip all lines that have a tag unclear

echo $inputdir
echo $outputdir
echo $filelist
echo $filelisttrain
echo $filelistval

find $inputdir -name '*.done' -exec rm {} \;
# run apptainr image to process the images, cutting them into small pieces
apptainer run tool.sif \
  /src/loghi-tooling/minions/target/appassembler/bin/MinionCutFromImageBasedOnPageXMLNew -input_path $inputdir -outputbase $outputdir -channels 4 -output_type png -write_text_contents -threads $numthreads $INCLUDETEXTSTYLES -no_page_update $SKIP_UNCLEAR -use_2013_namespace

echo "outputfiles: " `find $outputdir|wc -l`


count=0
> $filelist
for input_path in $(find $outputdir -name '*.png');
do
        filename=$(basename -- "$input_path")
        filename="${filename%.*}"
        base="${input_path%.*}"
        text=`cat $base.txt`
        echo -e "$input_path\t$text" >>$filelist
done

# suffule to generate training and validating set
shuf $filelist | split -l $(( $(wc -l <$filelist) * $trainsplit / 100 )); mv xab $filelistval; mv xaa $filelisttrain
