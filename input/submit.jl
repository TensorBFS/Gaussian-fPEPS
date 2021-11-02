workpath = "/home/qiyang/source/jaxgfpeps/jobs"

for i in cd(readdir,workpath)
    if i[end-2:end] == "pbs"
        cmd = ```qsub -V $workpath/$i```
        run(cmd)
        sleep(0.1)
    end
end