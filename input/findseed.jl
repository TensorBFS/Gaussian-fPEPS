Nv = ARGS[1]
JobID = ARGS[2]
dir = "/home/qiyang/source/jaxgfpeps/"
pythondir = "/home/qiyang/Downloads/Python3.8/bin/python"
L = 101
MuSet = [0.1*i for i = 1:9]

for Mu in MuSet
let
string = "#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -V
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -N BatchABDNv$(Nv)FSEEDS-$(JobID)Mu$(Mu)
#PBS -o /home/qiyang/source/jaxgfpeps/log/BatchABDNv$(Nv)FSEEDMu$(Mu).log
#PBS -e /home/qiyang/source/jaxgfpeps/log/BatchABDNv$(Nv)FSEEDMu$(Mu).log
"
for seednum in 4000:4010
    seed = seednum+4000*parse(Int,JobID)
string = string*"
$(pythondir) $(dir)/src/main.py --Mu $(Mu) --Lx $(L) --Ly $(L) --Nv $(Nv) --DeltaX 1.0 --DeltaY -1.0 --label $(seed)  --loadlabel $(seed) --seed $(seed)
"
end

io = open("$(dir)/jobs/AnealingABDNv$(Nv)FINDSEED-$(JobID)Mu$(Mu).pbs", "w")
println(io, string)
close(io)


end
end