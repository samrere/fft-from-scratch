## fft-from-scratch
This is a Python implementation of Fast Fourier Transform (FFT) in 1d and 2d from scratch and some of its applications in:
* Photo restoration (paper texture pattern removal)
* convolution (direct fft and overlap add fft method, including a comparison with the direct matrix multiplication method and ground truth using `scipy.signal.convolve`)
* large integer multiplication.

### [*Photo restoration*](https://nbviewer.org/github/samrere/fft-from-scratch/blob/main/pattern_removal.ipynb)
The honeycomb pattern on old photos is due to the "silk finish" paper texture, which was used a lot back in the day. We can remove them in frequency domain using fft.
#### view on larger screen, small screen will show aliasing effect. We can use a low pass filter later.
<p align="center">
  <img src="https://github.com/samrere/fft-from-scratch/blob/main/images/slider.gif" width="1000">
</p>

### [*Convolution*](https://nbviewer.org/github/samrere/fft-from-scratch/blob/main/convolution_comparison.ipynb)
achieves O(nlogn) complexity, whereas a direct matrix multiplication approach is O(n^2). It is more efficient when n is large.
	 
### [*large integer multiplication*](https://nbviewer.org/github/samrere/fft-from-scratch/blob/main/large_integer_multiplication.ipynb)
inspired from the multiplication of polynomials in chap.30 of Introduction to algorithms book.
