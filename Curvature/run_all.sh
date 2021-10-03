for i in `seq 0 1`
do
    for j in `seq 0 7`
    do
        for k in `seq 0 50`
            do /usr/local/bin/python3 subspace_curvature.py $i $j $k
        done
    done
done
