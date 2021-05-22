import mnist
from convolution import Convolution
from softmax import Softmax
from maxpool import MaxPooling
import numpy as np
train_images = mnist.train_images()

train_labels = mnist.train_labels()


convolution = Convolution(8)
pool=MaxPooling()
softmax=Softmax(13*13*8,10)
output = convolution.forwardPass(train_images[0])
output=pool.forward(output)
print(output.shape)  # the size  (26, 26, 8)
# just 1k example for testing purposes
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with. This is standard practice.
  out = convolution.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)
 
  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc


print('MNIST CNN initialized!')

loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
  # Do a forward pass.
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

  # Print stats every 100 steps.
  if i % 100 == 99:
    print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
    )
    loss = 0
    num_correct = 0
