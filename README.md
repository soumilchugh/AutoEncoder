# AutoEncoder implementation
 * Downsampling - Strided Convolution
 * Upsampling - Bilinear interpolation/ Transposed Convolution
 * Convolution - Depth Wise Convolution followed by a 1x1 convolution. Increases the speed by reducing the number of FLOPS
 * Skip connections between downsampling and upsampling blocks helps in graident flow
 * Latent Vector Size is 300
 * Loss - Mean Square Error
 * Optimizer - Adam
 * Can be used for denoising and compression
 * Layer Class contains the Convolution and Transposed Convolution Implementation
 * Data Class loads the data
 * Model class sets the loss and the optimizer
 * Train class runs the training and validation 
 
 # Input and their corresponding output images.
 * Here the input image has contrast and brightness enhanced. In addition salt and pepper noise is added in the form of small bright spots which are different from the real reflections on the cornea of the eye. 
 
 <img src="https://github.com/soumilchugh/AutoEncoder/blob/master/image1.jpg" height="300" width="200"> <img src=" https://github.com/soumilchugh/AutoEncoder/blob/master/image2.jpg" height="300" width="200"/> <img src=" https://github.com/soumilchugh/AutoEncoder/blob/master/image3.jpg" height="300" width="200"/> <img src="https://github.com/soumilchugh/AutoEncoder/blob/master/image4.png" height="300" width="200"/>
 
 # Output Images from the Autoencoder
 
 <img src="https://github.com/soumilchugh/AutoEncoder/blob/master/output1.jpg" height="300" width="200"> <img src=" https://github.com/soumilchugh/AutoEncoder/blob/master/output2.jpg" height="300" width="200"/> <img src=" https://github.com/soumilchugh/AutoEncoder/blob/master/output3.jpg" height="300" width="200"/> <img src="https://github.com/soumilchugh/AutoEncoder/blob/master/output4.png" height="300" width="200"/>

