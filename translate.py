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

def fiducial_hamiltonian(hρ,hκ):
    N = hρ.shape[0]
    dim = 2**N
    assert hρ.shape[1]==N and hκ.shape[1]==N and hκ.shape[1]==N
    assert np.linalg.norm(hρ-hρ.T.conj())<1E-10 and np.linalg.norm(hκ + hκ.T) < 1E-10
    
    H = np.zeros((2**N,2**N),dtype=np.complex128)

    for i in range(N):
        for j in range(N):
            for k in range(2**N):
                bk = bitarray.util.int2ba(k,length=N)
                bk.reverse()

                parity = (bk.count(1,i,N) + bk.count(1,j,N)) % 2
                
                if bk[i]==0:
                    if bk[j] == 1:
                        bk[i] = 1
                        bk[j] = 0
                        target = bitarray.util.ba2int(bk)
                        H[target,k] += hρ[i,j] * (-1)**parity
                    elif bk[j] == 0:
                        bk[j] = 1
                        bk[i] = 1
                        target = bitarray.util.ba2int(bk)
                        H[target,k] -= hκ[i,j].conj() * (-1)**parity    
    return H

def translate(Gamma,Nv):
    N = Gamma.shape[0]//2
    
    trans_h = cor_trans_matrix(-Gamma)
    
    hρ = -1.0j*trans_h[0:N,N:2*N].T
    hκ = 1.0j*trans_h[0:N,0:N]
    local_h = fiducial_hamiltonian(hρ,hκ)
    tw,tv = np.linalg.eig(local_h)
    return tv[:,0]

def paritygate(n):
    S = np.eye(n)
    for i in range(n):
        if bitarray.util.int2ba(i,int(np.ceil(np.log(n)/np.log(2)))).count() %2 !=0: 
            S[i,i] = -1
    return S

def fsign(n_list):
    result = 0
    for i in range(1,len(n_list)):
        result += n[i]*sum(n[0:i-1])
    return (-1)**(result % 2)

def bondgate(Nv):
    
    n = np.zeros((Nv,Nv)) # store n_i
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



def main(input_file):
    with h5py.File(input_file, "r") as fid:
        Nv = fid["/model/Nv"][()]
        T = fid["/transformer/T"][0:8*Nv+4,0:8*Nv+4]
        
    Gamma = getG(T,Nv)
    tensor_0 = translate(Gamma,Nv)
    
    assert abs(tensor_0[1])< 1E-10 # check parity
    tensor_1 = np.reshape(tensor_0,(2**Nv,2**Nv,4,2**Nv,2**Nv))

if __name__ == "__main__":
    main(input_file = "/home/yangqi/jaxgfpeps/data/default.h5")