#for i in {0..1}
#do
    for j in {0..7}
    do
        for k in {0..3}
        do
            #arg1=$i arg2=$j arg3=$k sbatch --job-name=curvature_analysis-$i-$j-$k curvature_experiments.sh
            arg1=1 arg2=$j arg3=$k sbatch --job-name=curvature_analysis-1-$j-$k curvature_experiments.sh
        done
    done
#done
