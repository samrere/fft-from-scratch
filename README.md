## fft-from-scratch
This is a Python implementation of Fast Fourier Transform (FFT) in 1d and 2d from scratch and some of its applications:
* Photo restoration (paper texture pattern removal)
* convolution (with speed comparison with the direct matrix multiplication method and ground truth using `scipy.signal.convolve`)

### [*Photo restoration*](https://nbviewer.org/github/samrere/fft-from-scratch/blob/main/pattern_removal.ipynb)
The honeycomb pattern on old photos is due to the "silk finish" paper texture, which was used a lot back in the day. We can remove them in frequency domain using fft.
before             |  after
:-------------------------:|:-------------------------:
![](https://github.com/samrere/fft-from-scratch/blob/main/images/old.jpg)  |  ![](https://github.com/samrere/fft-from-scratch/blob/main/images/new.jpg)

### [*Convolution*](https://nbviewer.org/github/samrere/fft-from-scratch/blob/main/convlution_comparison.ipynb)
achieves O(nlogn) complexity, whereas direct matrix multiplication approach is O(n^2). It is more efficient when n is large.
	 
