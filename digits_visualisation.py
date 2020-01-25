import matplotlib.pyplot as plt
import numpy as np
import gzip

f = gzip.open('train-images-idx3-ubyte.gz', 'r')
image_size = 28
count = 5

f.read(16)
buf = f.read(image_size * image_size * count)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(count, image_size, image_size, 1)

for i in range(0, count):
  image = np.asarray(data[i]).squeeze()
  plt.imshow(image)
  plt.show()
