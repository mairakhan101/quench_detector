#!/bin/bash

globalStartTime=$(date +%s)

for i in {24..57}
do
   iterationStartTime=$(date +%s)
   echo "scping ramp $i acoustics"
   scp -r maira@correlator3.FNAL.GOV:/data/maira/Acoustics/MBHSM03/rawData/Ramp${i}/* ~/Magnets/preprocess_data/Acoustics/
   echo "scping QU"
   scp -r maira@correlator3.FNAL.GOV:/data/maira/QA/MBHSM03/Ramp${i}/DAQU*/AllCards.*  ~/Magnets/preprocess_data/QA/QU/
   echo "scping QD"
   scp -r maira@correlator3.FNAL.GOV:/data/maira/QA/MBHSM03/Ramp${i}/DAQD*/AllCards.*  ~/Magnets/preprocess_data/QA/QD/
   echo "running python script for ramp $i"
   python pre_process.py $i
   echo "scping back ac_arr"
   scp Acoustics/ac_arr.npy maira@correlator3.FNAL.GOV:/data/maira/MagnetsPreProcessed/ac_arr_r${i}.npy
   echo "scping back t_arr"
   scp Acoustics/t_arr.npy maira@correlator3.FNAL.GOV:/data/maira/MagnetsPreProcessed/t_arr_r${i}.npy
   echo "scping back q_arr"
   scp QA/q_arr.npy maira@correlator3.FNAL.GOV:/data/maira/MagnetsPreProcessed/q_arr_r${i}.npy
   echo "rming ac_arr"
   rm Acoustics/ac_arr.npy
   echo "rming t_arr"
   rm Acoustics/t_arr.npy
   echo "rming q_arr"
   rm QA/q_arr.npy
   echo "rming acoustic data"
   rm Acoustics/*
   echo "rming QU data"
   rm QA/QU/*
   echo "rming QD data"
   rm QA/QD/*
   echo "done with output file $i"
   iterationEndTime=$(date +%s)
   iterationDuration=$((iterationEndTime - iterationStartTime))
   echo "iteration time = $iterationDuration"
done

globalEndTime=$(date +%s)

globalDuration=$((globalEndTime - globalStartTime))
echo "total time = $globalDuration"
