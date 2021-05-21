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



        
