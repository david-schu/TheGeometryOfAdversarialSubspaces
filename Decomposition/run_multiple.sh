for i in `seq 0 1`
do
    for j in `seq 0 24`
    do
        arg1=1 arg2=$i arg3=$j sbatch batch_run.sh        
    done
done
