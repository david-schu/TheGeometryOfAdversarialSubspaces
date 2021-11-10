for i in `seq 0 7`
do
    for j in `seq 41 50`
    do
        arg1=1 arg2=$i arg3=$j sbatch --job-name=curvature_analysis-$i-$j cifar_curvature.sh
    done
done
