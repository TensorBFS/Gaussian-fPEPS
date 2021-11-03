Nv = ARGS[1]
dir = "/home/qiyang/source/jaxgfpeps/"
pythondir = "/home/qiyang/Downloads/Python3.8/bin/python"
N = 40
L = 100
labels = [i for i in -N:N]
DXY =[1.0*10^(i/N) for i in -N:N]

string = "#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -V
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -N AnnealingABDNv$(Nv)
#PBS -o /home/qiyang/source/jaxgfpeps/log/AnnealingABDNv$(Nv).log
#PBS -e /home/qiyang/source/jaxgfpeps/log/AnnealingABDNv$(Nv).log

$(pythondir) $(dir)/src/main.py --Lx $(L) --Ly $(L) --MaxIter 10000 --Nv $(Nv) --DeltaX 1.0 --DeltaY -1.0 --label 0  --loadlabel 0

"
for i = 2+N:1:2N+1
    t = "$(pythondir) $(dir)/src/main.py --Lx $(L) --Ly $(L) --Nv $(Nv) --DeltaX $(DXY[i]) --DeltaY -$(DXY[i]) --label $(labels[i])  --loadlabel $(labels[i]-1)
"
    global string = string*t
end

string = string*"\n\n"

for i = N:-1:1
    t = "$(pythondir) $(dir)/src/main.py --Lx $(L) --Ly $(L) --Nv $(Nv) --DeltaX $(DXY[i]) --DeltaY -$(DXY[i]) --label $(labels[i])  --loadlabel $(labels[i]+1)
"
    global string = string*t
end

io = open("$(dir)/jobs/AnealingABDNv$(Nv).pbs", "w")
println(io, string)
close(io)

