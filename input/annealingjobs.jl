Nv = 1
dir = "/home/qiyang/source/jaxgfpeps/src/"

string = "#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -V
#PBS -l nodes=1:ppn=20:gpus=1
#PBS -N BatchABDNv$(Nv)
#PBS -o /home/qiyang/source/jaxgfpeps/log/BatchABDNv$(Nv).log
#PBS -e /home/qiyang/source/jaxgfpeps/log/BatchABDNv$(Nv).log

python $(dir)main.py --Lx 20 --Ly 20 --Nv $(Nv) --DeltaX 1.0 --DeltaY -1.0 --label 999 --loadlabel 999

"
N = 20

labels = [i for i in 0:20]
DXY =[1.0*10^(i/20) for i in 0:20]
for i = 1:length(labels)
run = "julia --project=/home/qiyang/source/fgpeps /home/qiyang/source/fgpeps/program/AnnealingABD/AnnealingABD.jl --Nv $(Nv) --DeltaX $(DXY[i]) --DeltaY -$(DXY[i]) --label $(labels[i]) --loadlabel $(labels[i]-1 == -1 ? 999 : labels[i]-1)
"
global string = string*run
end

labels = [i for i in -1:-1:-20]
DXY =[1.0*0.1^(i/20) for i = 1:20]
for i = 1:length(labels)
    run = "julia --project=/home/qiyang/source/fgpeps /home/qiyang/source/fgpeps/program/AnnealingABD/AnnealingABD.jl --Nv $(Nv) --DeltaX $(DXY[i]) --DeltaY -$(DXY[i]) --label $(labels[i]) --loadlabel $(labels[i]+1)
"
global string = string*run
end

io = open("/home/qiyang/source/fgpeps/program/AnnealingABD/jobs/AnealingABDNv$(Nv).pbs", "w")
println(io, string)
close(io)
