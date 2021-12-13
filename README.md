## fft-from-scratch
This is a Python implementation of Fast Fourier Transform (FFT) in 1d and 2d from scratch and some of its applications:
* Photo restoration (paper texture pattern removal)
* convolution (with speed comparison with the direct matrix multiplication method and ground truth using `scipy.signal.convolve`)

### Photo restoration
<p align="center">
  <img src="https://github.com/samrere/fft-from-scratch/blob/main/images/animation.gif" width="600">
</p>

### Convolution
achieves O(nlogn) complexity, whereas direct matrix multiplication approach is O(n^2). It is more efficient when n is large.
	 
