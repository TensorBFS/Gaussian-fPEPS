Nv = ARGS[1]
dir = "/home/qiyang/source/jaxgfpeps/"
pythondir = "/home/qiyang/Downloads/Python3.8/bin/python"
N = 20
LSet = [6;10;20;30;50;100]
labels = [i for i in -N:N]
DXY =[1.0*10^(i/N) for i in -N:N]

for i = 1:length(labels)
string = "#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -V
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -N BatchABDNv$(Nv)C$(labels[i])
#PBS -o /home/qiyang/source/jaxgfpeps/log/BatchABDNv$(Nv)C$(labels[i]).log
#PBS -e /home/qiyang/source/jaxgfpeps/log/BatchABDNv$(Nv)C$(labels[i]).log
"
for L in LSet
string = string*"
$(pythondir) $(dir)/src/main.py --Lx $(L) --Ly $(L) --Nv $(Nv) --DeltaX $(DXY[i]) --DeltaY -$(DXY[i]) --label $(labels[i])  --loadlabel $(labels[i]) 
"
end

io = open("$(dir)/jobs/AnealingABDNv$(Nv)C$(labels[i]).pbs", "w")
println(io, string)
close(io)
end


