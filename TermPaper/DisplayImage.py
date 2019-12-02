import matplotlib.pyplot as plt
from PIL import Image

im = Image.open("Pred-test-4-neuralnet.png")
plt.figure()
plt.imshow(im)
plt.show()
