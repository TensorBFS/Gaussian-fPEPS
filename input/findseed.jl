Nv = ARGS[1]
JobID = ARGS[2]
dir = "/home/qiyang/source/jaxgfpeps/"
pythondir = "/home/qiyang/Downloads/Python3.8/bin/python"
L = 101
Mu = 0.0
let

string = "#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -V
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -N BatchABDNv$(Nv)FSEEDS-$(JobID)
#PBS -o /home/qiyang/source/jaxgfpeps/log/BatchABDNv$(Nv)FSEED.log
#PBS -e /home/qiyang/source/jaxgfpeps/log/BatchABDNv$(Nv)FSEED.log
"
for seednum in 4000:4100
    seed = seednum+4000*parse(Int,JobID)
string = string*"
$(pythondir) $(dir)/src/main.py --Mu $(Mu) --Lx $(L) --Ly $(L) --Nv $(Nv) --DeltaX 1.0 --DeltaY -1.0 --label $(seed)  --loadlabel $(seed) --seed $(seed)
"
end

io = open("$(dir)/jobs/AnealingABDNv$(Nv)FINDSEED-$(JobID).pbs", "w")
println(io, string)
close(io)


end
