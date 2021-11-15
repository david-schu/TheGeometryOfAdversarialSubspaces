for j in {0..19} 
do
    arg1=$j sbatch --job-name=conv_$j conv.sh        
done
