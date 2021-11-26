seed=8112
CUDA=3
Nv=2

CUDA_VISIBLE_DEVICES=$CUDA /home/qiyang/bin/python /home/qiyang/source/jaxgfpeps/src/main.py --Mu 0.5 --Lx 101 --Ly 101  --DeltaX 1.0 --DeltaY -1.0 --label $seed  --loadlabel $seed --seed $seed --Nv $Nv --MaxIter 1000
mv /home/qiyang/source/jaxgfpeps/result/* /home/qiyang/source/gaussiantranslator/indir/
julia /home/qiyang/source/gaussiantranslator/run.jl
mv /home/qiyang/source/gaussiantranslator/outdir/* /home/qiyang/source/gaussiantranslator/data
mv /home/qiyang/source/gaussiantranslator/indir/* /home/qiyang/source/gaussiantranslator/data
