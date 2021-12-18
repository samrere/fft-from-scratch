import numpy as np

# DFT16=np.fromfunction(lambda x,y:np.exp(-2j*np.pi*x*y/16),shape=(16,16))
DFT16=np.exp(-2j*np.pi*np.arange(16)*np.arange(16)[:,None]/16) # faster
FACTOR={N: np.exp(-2j*np.pi*np.arange(N/2)/ N) for N in np.array([1024//32,1024//16,1024//8,1024//4,1024//2,
                                                                  1024,
                                                                  1024*2,1024*4,1024*8,1024*16,1024*32,1024*64,1024*128])}
def _dft(a,axis,forward):
    f=np.moveaxis(a,axis,-1)[...,None]
    N=f.shape[-2]
    DFT=np.exp((-1)**forward*2j*np.pi*np.arange(N)*np.arange(N)[:,None]/N)
    result=np.moveaxis((DFT@f)[...,-1],-1,axis)
    return result if forward else result/N
def _fft(a,axis,forward):
    N_min=16
    N=a.shape[axis]
    # check dimension
    if N&(N-1) != 0:
        raise ValueError("size of input must be a power of 2")
    # return using dft if dimension is small
    if N<=N_min:
        return _dft(a,axis,forward)
    # if not:
    x=np.moveaxis(a,axis,-1) # move axis to end
    # split
    DFT16_=DFT16 if forward else np.conjugate(DFT16)
    X = DFT16_@x.reshape((*x.shape[:-1],N_min,-1))
    # combine
    while X.shape[-2]<N:
        X_even=X[...,:X.shape[-1]//2]
        X_odd=X[...,X.shape[-1]//2:]
        factor=FACTOR[2*X.shape[-2]][...,None]
        factor_=factor if forward else np.conjugate(factor)
        t=factor_*X_odd
        X=np.concatenate((X_even+t,X_even-t),axis=-2)
    result=np.moveaxis(X[...,-1],-1,axis)
    return result if forward else result/N

def fft(a,axis=-1):
    return _fft(a,axis,True)
def ifft(a,axis=-1):
    return _fft(a,axis,False)

def fft2(a,axes=(-1,-2)):
    for axis in axes:
        a=_fft(a,axis,True)
    return a
def ifft2(a,axes=(-1,-2)):
    for axis in axes:
        a=_fft(a,axis,False)
    return a
