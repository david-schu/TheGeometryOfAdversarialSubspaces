for i in `seq 0 1`
do
    for j in `seq 0 49`
    do
        arg1=0 arg2=$i arg3=$j sbatch batch_run.sh        
    done
done
