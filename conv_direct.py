import numpy as np
def convolve_direct(img,kernel,mode='full'):
    assert mode in {'full','same','valid'}, NotImplemented
    kernel=np.flipud(np.fliplr(kernel))
    return cross_correlation(img,kernel,mode)

def cross_correlation(img, kernel,mode):
    if img.ndim==2:
        img=img[None,...]
    '''
    img: C*H*W, where C is channel (3 for rgb image, 1 for grayscale);
                  H and W are image height and width respectively.
    '''
    kernel_size = kernel.shape
    h_k, w_k = kernel.shape  # kernel height and width
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

    C, H_in, W_in = img.shape  # batch size, rgb channel, Height, Width
    H_out = np.floor(H_in + 2 * padding[0] - kernel_size[0] + 1).astype(int)
    W_out = np.floor(W_in + 2 * padding[1] - kernel_size[1] + 1).astype(int)

    expanded = np.lib.stride_tricks.as_strided(
        padded,
        shape=(
            H_out,  # out channel height
            W_out,  # out channel width
            padded.shape[-3],  # input channel
            kernel.shape[-2],  # kernel height
            kernel.shape[-1],  # kernel width
        ),
        strides=(
            padded.strides[-2],  # H dimension
            padded.strides[-1],  # W dimension
            padded.strides[-3],  # input chennel
            padded.strides[-2],  # kernel height
            padded.strides[-1],  # kernel width
        ),
        writeable=False,
    )
    feature_map = np.ascontiguousarray(np.moveaxis(np.einsum('...ij,...ij->...', expanded, kernel), -1, -3))
    return feature_map
