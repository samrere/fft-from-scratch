from fft import *

# direct matrix multiplication method
def convolve_direct(img,kernel,mode='full'):
    assert mode in {'full','same','valid'}, NotImplemented
    kernel=np.flip(kernel,(-1,-2))
    return cross_correlation(img,kernel,mode)

def cross_correlation(img, kernel,mode):
    '''
    img shape must be N,Cin,H,W: batch size, input channel, height and width
    kernel shape should be N,Cout,Hk,Wk: batch, output channel, height and width
    '''
    assert mode in {'full','same','valid'}, NotImplemented
    h_k, w_k = kernel.shape[-2:]  # kernel height and width
    if mode == 'full':
        padding = (h_k - 1, w_k - 1)
        pad_width = [[0, 0] for _ in range(img.ndim)]
        pad_width[-2] = [padding[0], padding[0]]
        pad_width[-1] = [padding[1], padding[1]]
        padded = np.pad(img, pad_width)
        return cross_correlation(padded, kernel, mode='valid')
    elif mode == 'same':
        padding = (h_k // 2, w_k // 2)
        pad_width = [[0, 0] for _ in range(img.ndim)]
        pad_width[-2] = [padding[0], padding[0]]
        pad_width[-1] = [padding[1], padding[1]]
        padded = np.pad(img, pad_width)
    elif mode == 'valid':
        padding = (0,0)
        padded=img

    N, C, H_in, W_in = img.shape  # batch size, rgb channel, Height, Width
    H_out = np.floor(H_in + 2 * padding[0] - h_k + 1).astype(int)
    W_out = np.floor(W_in + 2 * padding[1] - w_k + 1).astype(int)

    expanded = np.lib.stride_tricks.as_strided(
        padded,
        shape=(
            N,                 # batch
            H_out,             # out channel height
            W_out,             # out channel width
            kernel.shape[0],   # out channel
            padded.shape[-3],  # input channel
            kernel.shape[-2],  # kernel height
            kernel.shape[-1],  # kernel width
        ),
        strides=(
            padded.strides[0],   # batch
            padded.strides[-2],  # H dimension
            padded.strides[-1],  # W dimension
            0,                   # output channel
            padded.strides[-3],  # input channel
            padded.strides[-2],  # kernel height
            padded.strides[-1],  # kernel width
        ),
        writeable=False,
    )
    feature_map = np.ascontiguousarray(np.moveaxis(np.einsum('...ijk,...ijk->...', expanded, kernel), -1, -3))
    return feature_map

# fft convolution
def convolve_fft(image, kernel,mode='full'):
    assert mode in {'full','same','valid'}, NotImplemented

    h,w=image.shape[-2:]
    hk,wk=kernel.shape[-2:]
    # the shape of the output should be (h+hk-1, w+wk-1) for full convolution,
    # need to extend to next power of 2
    hf,wf=1<<(h+hk-2).bit_length(),1<<(w+wk-2).bit_length()
    
    pad_width = [[0, 0] for _ in range(image.ndim)]
    pad_width[-2] = [0,hf-h]
    pad_width[-1] = [0,wf-w]
    img_padded=np.pad(image,pad_width)
    pad_width = [[0, 0] for _ in range(kernel.ndim)]
    pad_width[-2] = [0,hf-hk]
    pad_width[-1] = [0,wf-wk]
    kernel_padded=np.pad(kernel,pad_width)
    image_hat=fft2(img_padded)
    kernel_hat=fft2(kernel_padded)
    output_hat=image_hat*kernel_hat
    
    
    result_fft=ifft2(output_hat)[...,:(h+hk-1), :(w+wk-1)]
    
    if np.isrealobj(image) and np.isrealobj(kernel):
        result_fft = np.real(result_fft)
    if mode=='same':
        return result_fft[...,(hk//2):(h+hk//2),(wk//2):(w+wk//2)]
    elif mode=='valid':
        return result_fft[...,(hk-1):(hk-1+h-2*(hk//2)),(wk-1):(wk-1+w-2*(wk//2))]
    elif mode=='full':
        return result_fft


# overlap add fft
def convolve_oa(img, kernel,mode='full',chunk_thres=[1024,1024]):
    assert mode in {'full','same','valid'}, NotImplemented
    

    if img.ndim==2:
        img=img[None,...]

    chunk_thres=np.array(chunk_thres)
    chunk_size=np.minimum(chunk_thres+1-kernel.shape,np.array(img.shape[-2:]))
    if chunk_size[0]<0 or chunk_size[1]<0:
        raise ValueError(f'chunk_thres{chunk_thres} should be larger than kernel size{kernel.shape} in all dimensions.')
    hc,wc=chunk_size
    chunk_overlap=chunk_thres-chunk_size

    hi,wi=img.shape[-2:]
    hk,wk=kernel.shape[-2:]
    output=np.zeros((*img.shape[:-2],hi+hk-1,wi+wk-1))
    HH,WW=output.shape[-2:]
    H_out,W_out=img.shape[-2:]//chunk_size

    start_h=0
    for h in range(H_out+1):
        start_w=0
        for w in range(W_out+1):
            part=img[...,(h*hc):((h+1)*hc),(w*wc):((w+1)*wc)]
            if part.size:
                result=convolve_fft(part,kernel)
                hr,wr=result.shape[-2:]

                end_h=min(start_h+hr,HH)
                end_w=min(start_w+wr,WW)
                output[...,start_h:end_h,start_w:end_w]+=result[...,:(end_h-start_h),:(end_w-start_w)]
            start_w+=wr-chunk_overlap[1]
        start_h+=hr-chunk_overlap[0]
    
    if mode=='same':
        return output[...,(hk//2):(hi+hk//2),(wk//2):(wi+wk//2)]
    elif mode=='valid':
        return output[...,(hk-1):(hk-1+hi-2*(hk//2)),(wk-1):(wk-1+wi-2*(wk//2))]
    elif mode=='full':
        return output


