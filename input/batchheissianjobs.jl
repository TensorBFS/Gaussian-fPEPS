Nv = ARGS[1]
dir = "/home/qiyang/source/jaxgfpeps/"
pythondir = "/home/qiyang/Downloads/Python3.8/bin/python"
N = 20
labels = [i for i in -N:N]
DXY =[1.0*10^(i/N) for i in -N:N]
L = 20

for i = 1:length(labels)
string = "#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -V
#PBS -l nodes=1:ppn=20:gpus=1
#PBS -N BatchHessianABDNv$(Nv)C$(labels[i])
#PBS -o /home/qiyang/source/jaxgfpeps/log/BatchHessianABDNv$(Nv)C$(labels[i]).log
#PBS -e /home/qiyang/source/jaxgfpeps/log/BatchHessianABDNv$(Nv)C$(labels[i]).log

$(pythondir) $(dir)/src/mainhessian.py --Lx $(L) --Ly $(L) --Nv $(Nv) --DeltaX $(DXY[i]) --DeltaY -$(DXY[i]) --label $(labels[i])  --loadlabel $(labels[i]) 
"

io = open("$(dir)/jobs/AnealingHessianABDNv$(Nv)C$(labels[i]).pbs", "w")
println(io, string)
close(io)
end

