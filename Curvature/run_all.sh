for i in `seq 0 1`
do
    for j in `seq 0 7`
    do
        for k in `seq 0 49`
        do
            arg1=$i arg2=$j arg3=$k sbatch --job-name=curvature_analysis-$i-$j-$k cifar_curvature.sh
        done
    done
done
