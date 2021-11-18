#for i in {0..1}
#do
    for j in {0..7}
    do
        for k in {7..9}
        do
            #arg1=$i arg2=$j arg3=$k sbatch --job-name=curvature_analysis-$i-$j-$k curvature_experiments.sh
            JOB_NAME=curvature_analysis-1-$j-$k
            IS_RUNNING=$(squeue --name="$JOB_NAME" --noheader)
            if [ -z "$IS_RUNNING" ]; then
                echo "starting $JOB_NAME"
                arg1=1 arg2=$j arg3=$k sbatch --job-name=$JOB_NAME curvature_experiments.sh
            else
                echo "already running $JOB_NAME"
            fi
        done
    done
#done
