for j in {51..99}
do
    arg1=1 arg2=1 arg3=$j sbatch --job-name=decomp_1_$j batch_run.sh        
done
