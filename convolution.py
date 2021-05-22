import numpy as np

class Convolution:#3x3 tensor
    """
        We Would Work with a convolution layer using 3*3 filters. 
    """
    def __init__(self,number_of_filters):
        self.number_of_filters=number_of_filters
        """
          filters or kernel is a 3d array with dimensions being in this case (number_of_filters, 3, 3)
          We divide by 9 to reduce the variance of our initial values
          this is due to Something called Xavier Initialization for the training to be effective
         """
        self.filters = np.random.randn(number_of_filters, 3, 3)  #say numberof_filter=3
        # i will probably use a number of filter somthing thta ranges between 8 and 10 for THE INITIAL LAYER TO TRAIN THE MNIST DATASET
        # IF USED 8

        """the array is going take this format  it s what known as a tensor 
        [
            [
                [ 1.9609643  -1.89882763 23]                       
                [ 0.52252173  0.08159455 4543]
                [ 0.52252173  0.08159455 4543]
            ]
            [
                [-0.6060213  -0.86759247 153]
                [ 0.53870235 -0.77388125 5415]
                [ 0.52252173  0.08159455 4543]
            ]
            
            [
                [-0.6060213  -0.86759247 153]
                [ 0.53870235 -0.77388125 5415]
                [ 0.52252173  0.08159455 4543]
            ]

        ]
        """
        #now due to what's known as the Xavier Initialization
        self.filters/=9
        #WILL IMPLEMENT THE FEED FOWRAD LAYERS PORTION AT FIRST
    def iterate_regions(self,image):#image is assumed a 2d numpy array
        """ the soul purpose of this method is to generate all 3*3 image regions with valid padding
        """
        h,w=image.shape
        #since the padding offset with 2 
        for i in range(h - 2):
            for j in range(w - 2):
                region = image[i:(i + 3), j:(j + 3)]#to extract only a small the portion of 3x3 to run it through the network
                yield region, i, j

    def forwardPass(self,input):
        #iput is a 2d array/matrix
        ''' should make a forward pass of the convolutional layer and returns a 3d '''
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.number_of_filters))

        for region, i, j in self.iterate_regions(input):
            # this is exactly when convolution happens
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))
        return output



        
