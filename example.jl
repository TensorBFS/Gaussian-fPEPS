### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ eb82c5d3-5780-47cb-a96f-38d024264f48
using ITensors,OMEinsum,LinearAlgebra,HDF5,BitBasis

# ╔═╡ f9e2d306-12f7-11ed-30b3-eb4d4e324b10
md"""Here is an example of full workflow.
First, we need to obtain a optim $X$ in Eq.(12).


We have to execute the fgpeps/main.py. The default output file is in ./data"""

# ╔═╡ 2ed497f7-e407-471b-9d4a-34ba0f74f74c
md"""Then run the main.py... GPU is required as default. This is an example
```bash
srun --pty --partition=a100 --gres=gpu:A100_40G:1 ipython fgpeps/main.py
```
After optimization, we can check the setup and final $X$ with h5dump

```bash
h5dump ./data/default.h5
```
Then we can begin the translation:
"""

# ╔═╡ 99f8eb4b-52b2-40d5-98fd-77f1f0589552
skew(x) = x-transpose(x);

# ╔═╡ 8526877a-902b-4e7c-b7f2-4afa1f9da0ff
"""
	Permute G respect to the order of fermions
"""
function permuteG(G::Array,Nv::Int)
	function permutation_order(Nv::Int)
	    if Nv == 1
	        return [9, 7, 1, 2,11, 5,10, 8, 3, 4,12, 6]
	    elseif Nv == 2
	        return [13,17, 7,11, 1, 2,15,19, 5, 9,14,18, 8,12, 3, 4,16,20, 6,10]
	    elseif Nv == 3
	        return [17,21,25, 7,11,15, 1, 2,19,23,27, 5, 9,13,18,22,26, 8,12,16, 3, 4,20,24,28, 6,10,14]
	    elseif Nv == 4
	 	return [21, 25, 29, 33,  7, 11, 15, 19,  1,  2, 23, 27, 31, 35,  5,  9, 13, 17, 22, 26, 30, 34,  8, 12, 16, 20,  3,  4, 24, 28, 32, 36,  6, 10, 14, 18]
	    elseif Nv == 5
		return [25, 29, 33, 37, 41, 7, 11, 15, 19, 23, 1, 2, 27, 31, 35, 39, 43, 5, 9, 13, 17, 21, 26, 30, 34, 38, 42, 8, 12, 16, 20, 24, 3, 4, 28, 32, 36, 40, 44, 6, 10, 14, 18, 22]
	    else
	        print("Warn: perm for Nv=$(Nv) is not implemented!\n")
	        return [0]
	    end
	end

	function permutation_matrix(Nv::Int)
	    perm = permutation_order(Nv)
	    M = zeros(length(perm),length(perm))
	    for i = 1:length(perm)
	        M[perm[i],i] = 1.0
	    end
	    return M
	end
	
	P = permutation_matrix(Nv)

	
    return transpose(P)*G*P
end

# ╔═╡ 4a9d3cd6-7d12-4af9-b114-e73920883804
"""
	Get correlation matrix from T in Eq.(12)
"""
function getG(T::Array,Nv)
	J(Nv::Int) = skew(reshape([i-j==1 && mod(i,2)==0 for i=1:8Nv+4 for j=1:8Nv+4],8Nv+4,8Nv+4))
	
    optim_G = transpose(T)*J(Nv)*T
    return permuteG(optim_G,Nv)
end

# ╔═╡ 7c0683f0-70c6-4ccf-aa11-6fc76babf0eb
"""
	Convert a Correlation matrix of Majorana fermions back to complex fermions
"""
function cor_trans_matrix(cm::Matrix)
    N = size(cm)[1]÷2
    one = Matrix{ComplexF64}(I,N,N)
    S = [one one;+one*im -one*im]
    return transpose(S)*cm*S
end

# ╔═╡ f30cb874-2a46-4b91-90ea-5894e111b7c8
"""
	Construct fiducial hamiltonian in Eq.(13)
"""
function fiducial_hamiltonian(hρ::Array,hκ::Array)
    N = size(hρ)[1]
    @assert size(hρ)[2]==N && size(hκ)[2]==N && size(hκ)[2]==N
    @assert norm(hρ-adjoint(hρ))<1E-10 && norm(hκ+transpose(hκ))<1E-10
    
    ampo = AutoMPO()
    sites = siteinds("Fermion",N)
    for i  = 1:N
        for j = 1:N
            ampo += 2*hρ[i,j], "Cdag",i,"C",j
            ampo += hκ[i,j], "C",i,"C",j
            ampo += conj(hκ[i,j]), "Cdag",j,"Cdag",i
        end
    end
    H = MPO(ampo,sites)
    op = Array(H[1],inds(H[1])...)
    for j = 2:N-1
        opa = Array(H[j],inds(H[j])...)
        op = reshape(ein"aij,abpq->bipjq"(op,opa),:,2^j,2^j)
    end
    opn = Array(H[N],inds(H[N])...)
    op = reshape(ein"aij,apq->ipjq"(op,opn),2^N,2^N)

    @assert norm(op-adjoint(op)) <1E-10
    return Hermitian(op)
end

# ╔═╡ c7ae626e-3419-4891-b9b6-571ac92123c4
"""
	Translate: from Gamma to local tensor in METHOD.SETCTION(II.C)
"""
function translate(Gamma::Array,Nv::Int)
    N = size(Gamma)[1]÷2
	
    trans_h = cor_trans_matrix(-Gamma)
    
    hρ = -im*transpose(trans_h[1:N,N+1:2N])
    hκ = im*trans_h[1:N,1:N]
    local_h = fiducial_hamiltonian(hρ,hκ)
    tw,tv = eigen(local_h)
    return tv[:,1]
end

# ╔═╡ 3dde4a29-1e28-473d-a88d-b63638c9f5d8
input_file = "/home/yangqi/jaxgfpeps/data/default.h5";

# ╔═╡ 5f6214ea-7200-4241-9eb1-02101551949d
begin
	fid = h5open(input_file, "cw");
	Nv = fid["/model/Nv"][]
	T = fid["/transformer/T"][1:8Nv+4,1:8Nv+4]
	T = permutedims(T,(2,1)) # Python Julia notation transform
	close(fid)
	md"""read  hdf5 file"""
end

# ╔═╡ 15df8987-52f9-47b0-8643-255d271db509
begin
	Gamma = getG(T,Nv)
	md"""get Gamma"""
end

# ╔═╡ 3962f6a6-c48f-4777-83f0-56e99c92dc0a
tensor_0 = translate(Gamma,Nv);

# ╔═╡ c02fe562-4176-43dc-9825-8a9a1252b085
md"""It is tedious to deal odd parity tensor, we can check if the parity is odd. If so, we may need to restart the first step: optimization."""

# ╔═╡ 02d089c1-dafd-4b78-9100-d9574cc50b42
abs(tensor_0[1])< 1E-10

# ╔═╡ dca8528a-4ea1-45cb-be29-9a53a1fdf6f8
#Reshape it
tensor_1 = reshape(tensor_0,2^Nv,2^Nv,4,2^Nv,2^Nv);

# ╔═╡ fb8a9173-20c9-4aff-a75a-f23220f146ce
begin
	
"""
	function paritygate(n::Int)
		return a parity gate.(Matrix)

Example:
julia> paritygate(4)
4×4 Matrix{Float64}:
 1.0   0.0   0.0  0.0
 0.0  -1.0   0.0  0.0
 0.0   0.0  -1.0  0.0
 0.0   0.0   0.0  1.0
"""
function paritygate(n::Int)
	S = Matrix{Float64}(I,n,n)
	for i = 1:n
		if sum(bitarray(i-1,Int(ceil(log(n)/log(2)))))%2 !=0 
			S[i,i] = -1
		end
	end
	return S
end


"""
	each bond exist a bond which is responsible for exchange of virtual complex fermions.
		
		julia> bondgate(2)
		4×4 Matrix{Float64}:
		 1.0  0.0  0.0   0.0
		 0.0  1.0  0.0   0.0
		 0.0  0.0  1.0   0.0
		 0.0  0.0  0.0  -1.0
		
		julia> bondgate(3)
		8×8 Matrix{Float64}:
		 1.0  0.0  0.0   0.0  0.0   0.0   0.0   0.0
		 0.0  1.0  0.0   0.0  0.0   0.0   0.0   0.0
		 0.0  0.0  1.0   0.0  0.0   0.0   0.0   0.0
		 0.0  0.0  0.0  -1.0  0.0   0.0   0.0   0.0
		 0.0  0.0  0.0   0.0  1.0   0.0   0.0   0.0
		 0.0  0.0  0.0   0.0  0.0  -1.0   0.0   0.0
		 0.0  0.0  0.0   0.0  0.0   0.0  -1.0   0.0
		 0.0  0.0  0.0   0.0  0.0   0.0   0.0  -1.0
"""
function bondgate(Nv::Int)
	ind = CartesianIndices(Tuple([1:2 for i =1:Nv]))
    n = zeros(Int,Nv) # store n_i
    p = zeros([2 for i =1:Nv]...)
    for index in ind
        for i = 1:Nv
            n[i] = Tuple(index)[i]
        end
        n = n.-1
        p[index] = fsign(n)
    end
    return Array(Diagonal(p[:]))
end

"""
	fsign(n) = n1*(n2+n3+n4...) + n2(n3+n4...)
	coming from exchange complex fermions on each bond.
"""
function fsign(n::Array{Int})
    result = 0
    for i = 2:length(n)
        result += n[i]*sum(n[1:i-1])
    end
    return (-1)^mod(result,2)
end

"""
	function add_bondgate(T::Array,dim::N,Nv::Int)	

		add an bondgate for T at dim with Nv
"""
function add_bondgate(T::Array,dim::Int,Nv::Int)
	s = size(T)
	@assert s[dim]==2^Nv
	
	perm = collect(1:length(s))
	perm[dim] = 1
	perm[1] = dim

	T = permutedims(T,perm)
	T = reshape(T,s[dim],:)
	T = permutedims(reshape(ein"ij,io->oj"(T,bondgate(Nv)),s[perm]),perm)
	return T
end

"""
	function add_bondgate(T::Array,dim::N,Nv::Int)	

		add an bondgate for T at dim with Nv
"""
function add_paritygate(T::Array,dim::Int,Nv::Int)
	s = size(T)
	@assert s[dim]==2^Nv
	
	perm = collect(1:length(s))
	perm[dim] = 1
	perm[1] = dim

	T = permutedims(T,perm)
	T = reshape(T,s[dim],:)
	T = permutedims(reshape(ein"ij,io->oj"(T,paritygate(2^Nv)),s[perm]),perm)
	return T
end

	md"""Here we omit some functions about swapgates"""
end

# ╔═╡ 66c7861b-4eb3-46b1-8852-e12925763f94
"""
	Add swap gates to the tensor.
"""
function add_gates(tensor)
	tensor = add_bondgate(tensor,1,Nv) # bond gate in Eq.(16). on u
	tensor = add_bondgate(tensor,2,Nv) # bond gate in Eq.(16). on l
	tensor = add_paritygate(tensor,1,Nv) # As ulfdr  vs  (ud-rl in ABD)
	return tensor
end

# ╔═╡ 67adfc29-0a86-4464-83c7-09c386a0ed11
tensor_final = add_gates(tensor_1)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BitBasis = "50ba71b6-fa0f-514d-ae9a-0916efc90dcf"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
ITensors = "9136182c-28ba-11e9-034c-db9fb085ebd5"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"

[compat]
BitBasis = "~0.8.1"
HDF5 = "~0.16.10"
ITensors = "~0.3.18"
OMEinsum = "~0.7.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "5c0b629df8a5566a06f5fef5100b53ea56e465a0"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BatchedRoutines]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "441db9f0399bcfb4eeb8b891a6b03f7acc5dc731"
uuid = "a9ab73d0-e05c-5df1-8fde-d6a4645b8d8e"
version = "0.2.2"

[[deps.BetterExp]]
git-tree-sha1 = "dd3448f3d5b2664db7eceeec5f744535ce6e759b"
uuid = "7cffe744-45fd-4178-b173-cf893948b8b7"
version = "0.1.0"

[[deps.BitBasis]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f51ef0fdfa5d8643fb1c12df3899940fc8cf2bf4"
uuid = "50ba71b6-fa0f-514d-ae9a-0916efc90dcf"
version = "0.8.1"

[[deps.BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "5a814467bda636f3dde5c4ef83c30dd0a19928e0"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.2.6"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "49549e2c28ffb9cc77b3689dc10e46e6271e9452"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.12.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Dictionaries]]
deps = ["Indexing", "Random"]
git-tree-sha1 = "36bc84c68847edd2a3f97f32839fa484d1e1bce7"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.22"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "73145f1d724b5ee0e90098aec39a65e9697429a6"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.4.2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "d88b17a38322e153c519f5a9ed8d91e9baa03d8f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.1"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "1067cd05184719ba86f19cf1d49d57f0bcbabbf6"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.16.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "9ffc57b9bb643bf3fce34f3daf9ff506ed2d8b7a"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.10"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "c003b31e2e818bc512b0ff99d7dce03b0c1359f5"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+1"

[[deps.ITensors]]
deps = ["BitIntegers", "ChainRulesCore", "Compat", "Dictionaries", "HDF5", "IsApprox", "KrylovKit", "LinearAlgebra", "LinearMaps", "NDTensors", "PackageCompiler", "Pkg", "Printf", "Random", "Requires", "SerializedElementArrays", "StaticArrays", "Strided", "TimerOutputs", "TupleTools", "Zeros", "ZygoteRules"]
git-tree-sha1 = "4a2abd9e21919d8991a089f30347f2bb1bf51eda"
uuid = "9136182c-28ba-11e9-034c-db9fb085ebd5"
version = "0.3.18"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IsApprox]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fcf3bcf04bea6483b9d0aa95cef3963ffb4281be"
uuid = "28f27b66-4bd8-47e7-9110-e2746eb8bed7"
version = "0.1.4"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.KrylovKit]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "49b0c1dd5c292870577b8f58c51072bd558febb9"
uuid = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
version = "0.5.4"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "d1b46faefb7c2f48fdec69e6f3cc34857769bc15"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.8.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NDTensors]]
deps = ["Compat", "Dictionaries", "HDF5", "LinearAlgebra", "Random", "Requires", "StaticArrays", "Strided", "TimerOutputs", "TupleTools"]
git-tree-sha1 = "7ee25b3cead37da78b5c4a25ee5dee0834321a93"
uuid = "23ae76d9-e61a-49c4-8f12-3f1a16adf9cf"
version = "0.1.42"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OMEinsum]]
deps = ["AbstractTrees", "BatchedRoutines", "CUDA", "ChainRulesCore", "Combinatorics", "LinearAlgebra", "MacroTools", "OMEinsumContractionOrders", "Requires", "Test", "TupleTools"]
git-tree-sha1 = "ae3d05fe63984bab61cffa513544d5acd178c013"
uuid = "ebe7aa44-baf0-506c-a96f-8464559b3922"
version = "0.7.1"

[[deps.OMEinsumContractionOrders]]
deps = ["AbstractTrees", "BetterExp", "JSON", "Requires", "SparseArrays", "Suppressor"]
git-tree-sha1 = "d1efdca5b4556689d115f44b7039f32300379f1c"
uuid = "6f22d1fd-8eed-4bb7-9776-e7d684900715"
version = "0.7.1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.PackageCompiler]]
deps = ["Artifacts", "LazyArtifacts", "Libdl", "Pkg", "Printf", "RelocatableFolders", "TOML", "UUIDs"]
git-tree-sha1 = "52ab501c2201e140954924365ef06a36ba1d97ed"
uuid = "9b87118b-4619-50d2-8e1e-99f35a4d4d9d"
version = "2.0.7"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "7a1a306b72cfa60634f03a911405f4e64d1b718b"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SerializedElementArrays]]
deps = ["Serialization"]
git-tree-sha1 = "8e73e49eaebf73486446a3c1eede403bff259826"
uuid = "d3ce8812-9567-47e9-a7b5-65a6d70a3065"
version = "0.1.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "23368a3313d12a2326ad0035f0db0c0966f438ef"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "a7a664c91104329c88222aa20264e1a05b6ad138"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.3"

[[deps.Suppressor]]
git-tree-sha1 = "c6ed566db2fe3931292865b966d6d140b7ef32a9"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "464d64b2510a25e6efe410e7edab14fffdc333df"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.20"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zeros]]
deps = ["Test"]
git-tree-sha1 = "7eb4fd47c304c078425bf57da99a56606150d7d4"
uuid = "bd1ec220-6eb4-527a-9b49-e79c3db6233b"
version = "0.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─f9e2d306-12f7-11ed-30b3-eb4d4e324b10
# ╟─2ed497f7-e407-471b-9d4a-34ba0f74f74c
# ╠═eb82c5d3-5780-47cb-a96f-38d024264f48
# ╠═99f8eb4b-52b2-40d5-98fd-77f1f0589552
# ╟─4a9d3cd6-7d12-4af9-b114-e73920883804
# ╟─8526877a-902b-4e7c-b7f2-4afa1f9da0ff
# ╟─7c0683f0-70c6-4ccf-aa11-6fc76babf0eb
# ╟─f30cb874-2a46-4b91-90ea-5894e111b7c8
# ╟─c7ae626e-3419-4891-b9b6-571ac92123c4
# ╠═3dde4a29-1e28-473d-a88d-b63638c9f5d8
# ╠═5f6214ea-7200-4241-9eb1-02101551949d
# ╠═15df8987-52f9-47b0-8643-255d271db509
# ╠═3962f6a6-c48f-4777-83f0-56e99c92dc0a
# ╟─c02fe562-4176-43dc-9825-8a9a1252b085
# ╠═02d089c1-dafd-4b78-9100-d9574cc50b42
# ╠═dca8528a-4ea1-45cb-be29-9a53a1fdf6f8
# ╟─fb8a9173-20c9-4aff-a75a-f23220f146ce
# ╟─66c7861b-4eb3-46b1-8852-e12925763f94
# ╠═67adfc29-0a86-4464-83c7-09c386a0ed11
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
