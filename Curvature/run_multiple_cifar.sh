for i in `seq 0 7`
do
    for j in `seq 0 49`
    do
        arg1=1 arg2=$i arg3=$j sbatch cifar_curvature.sh
    done
done
