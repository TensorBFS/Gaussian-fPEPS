import jax.numpy as jnp
import jax
import numpy as np
jax.config.update("jax_enable_x64", True)
import logging
import h5py
import bitarray
import bitarray.util

def skew(x): return x-x.T

def permuteG(G,Nv):
    def permutation_order(Nv):
        match Nv:
            case 1:
                order = [8 ,6 , 0,  1, 10,  4,  9,  7,  2,  3, 11,  5]
            case 2:
                order = [12, 16,  6, 10,  0,  1, 14, 18,  4,  8, 13, 17,  7, 11,  2,  3, 15, 19,  5,  9]
            case 3:
                order = [16, 20, 24,  6, 10, 14,  0,  1, 18, 22, 26,  4,  8, 12, 17, 21, 25, 7, 11, 15,  2,  3, 19, 23, 27,  5,  9, 13]
            case 4:
                order = [20, 24, 28, 32,  6, 10, 14, 18,  0,  1, 22, 26, 30, 34,  4,  8, 12, 16, 21, 25, 29, 33,  7, 11, 15, 19,  2,  3, 23, 27, 31, 35,  5,  9,13, 17]
            case 5:
                order = [24, 28, 32, 36, 40,  6, 10, 14, 18, 22,  0,  1,  26, 30, 34, 38, 42,  4,  8, 12, 16, 20, 25, 29, 33, 37, 41,  7, 11, 15, 19, 23,  2,  3,27, 31, 35, 39, 43,  5,  9, 13, 17, 21]
            case _:
                raise ValueError("Nv must be 1,2,3,4,5")
        return np.array(order)


    def permutation_matrix(Nv):
        perm = permutation_order(Nv)
        M = np.zeros((len(perm),len(perm)))
        for i in range(len(perm)):
            M[perm[i],i] = 1.0
        return M
    
    P = permutation_matrix(Nv)
    return P.T @ G @ P

def getG(T, Nv):
    def J(Nv): return skew(np.reshape(np.array([int(j-i==1 and np.mod(j+1,2)==0) for i in range(8*Nv+4) for j in range(8*Nv+4)]),(8*Nv+4,8*Nv+4)))
    
    optim_G = T.T @ J(Nv) @T
    return permuteG(optim_G,Nv)

def cor_trans_matrix(cm):
    N = cm.shape[0] // 2
    one = np.eye(N)
    S = np.block([[one,one],[+one*1.0j ,-one*1.0j]])
    return S.T @ cm @ S

# Tested, respect to the original Julia code
def fiducial_hamiltonian(hρ,hκ):
    N = hρ.shape[0]
    
    # symmetry of hρ and hκ
    assert hρ.shape[1]==N and hκ.shape[1]==N and hκ.shape[1]==N
    assert np.linalg.norm(hρ-hρ.T.conj())<1E-10 and np.linalg.norm(hκ + hκ.T) < 1E-10
    
    H = np.zeros((2**N,2**N),dtype=np.complex128)
    
    for i in range(N):
        for j in range(N):
            for k in range(2**N):
                bk = bitarray.util.int2ba(k,length=N)
                bk.reverse()
                parity = (bk.count(1,i+1,N) + bk.count(1,j+1,N)) % 2
                if bk[j]==1:
                    if bk[i] == 0 or i == j:
                        bk[j] = 0
                        bk[i] = 1
                        bk.reverse()
                        target = bitarray.util.ba2int(bk)
                        parity += int(j > i)
                        H[target,k] += 2 * hρ[i,j] * (-1)**parity
                    elif bk[i] == 1:
                        bk[i] = 0
                        bk[j] = 0
                        bk.reverse()
                        target = bitarray.util.ba2int(bk)
                        parity += int(i > j)
                        H[target,k] += hκ[i,j] * (-1)**parity
                elif bk[j]==0:
                    if bk[i]==0:
                        bk[j] = 1
                        bk[i] = 1
                        bk.reverse()
                        target = bitarray.util.ba2int(bk)
                        parity += int(i > j)
                        H[target,k] -= hκ[i,j].conj() * (-1)**parity

    return (H + H.T.conj())/2

def translate(Gamma):
    assert Gamma.shape[0] % 2 == 0
    N = Gamma.shape[0]//2
    
    trans_h = cor_trans_matrix(-Gamma)
    
    hρ = -1.0j*trans_h[0:N,N:2*N].T
    hκ = 1.0j*trans_h[0:N,0:N]
    local_h = fiducial_hamiltonian(hρ,hκ)
    tw,tv = np.linalg.eigh(local_h) # eigh to search lowest eigenvalue
    return tv[:,0]

def paritygate(Nv):
    n = 2**Nv
    S = np.eye(n)
    for i in range(n):
        if bitarray.util.int2ba(i,int(np.ceil(np.log(n)/np.log(2)))).count() %2 !=0: 
            S[i,i] = -1
    return S

def fsign(n_list):
    result = 0
    for i in range(1,len(n_list)):
        result += n_list[i]*sum(n_list[0:i])
    return (-1)**(result % 2)

def bondgate(Nv):
    p = np.zeros(2**Nv)
    for i in range(2**Nv):
        n_list = list(map(int,bitarray.util.int2ba(i,Nv).to01()))
        p[i] = fsign(n_list)
    return np.diag(p)

def add_gates(tensor,Nv):
    
    tensor = np.einsum("ulfdr,iu->ilfdr",tensor, bondgate(Nv)) # bond gate in Eq.(16). on u
    tensor = np.einsum("ulfdr,il->uifdr",tensor, bondgate(Nv)) # bond gate in Eq.(16). on l
    tensor = np.einsum("ulfdr,iu->ilfdr",tensor, paritygate(Nv)) # As ulfdr  vs  (ud-rl in ABD)

    return tensor

def main(input_file):
    with h5py.File(input_file, "r") as fid:
        Nv = int(fid["/model/Nv"][()])
        T = fid["/transformer/T"][0:8*Nv+4,0:8*Nv+4]
        
    Gamma = getG(T,Nv)
    tensor_0 = translate(Gamma)
    
    assert abs(tensor_0[0])< 1E-10 # check parity 
    tensor_1 = np.reshape(tensor_0,(2**Nv,2**Nv,4,2**Nv,2**Nv)).transpose(4,3,2,1,0) # orderf of this reshape
    tensor_final = add_gates(tensor_1,Nv)
    return tensor_final

if __name__ == "__main__":
    np.set_printoptions(precision=6)
    input_file = "/home/yangqi/jaxgfpeps/data/default.h5"
    tensor = main(input_file)
    
    with h5py.File("tensor.h5", "cw") as fid:
        fid.create_dataset("/tensor", data=tensor) # order: ulfdr
    
    # fac = 0.396379-0.918087j # coefficient to match the original code
