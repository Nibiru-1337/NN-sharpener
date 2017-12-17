import numpy as np
import math

from PIL import Image


def gkern(shape=(3,3), sigma=3):
    """Returns a 2D Gaussian kernel array."""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gkern2(size, radius=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / radius**2)

def blur(a):
    kernel = np.array([[1.0,4.0,1.0], [4.0,16.0,4.0], [1.0,4.0,1.0]])
    kernel = kernel / np.sum(kernel)
    #kernel = gkern2(4, 5)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

def get_image_chunks(image, chunk_size=(8,8)):
    x_range = image.shape[0] // chunk_size[0]
    y_range = image.shape[1] // chunk_size[1]
    chunks = np.zeros(shape=(x_range, y_range, chunk_size[0], chunk_size[1], 3))
    for x in range(x_range):
        for y in range(y_range):
            x_start = x*chunk_size[0]
            y_start = y*chunk_size[1]
            chunks[x, y] = image[x_start:x_start+chunk_size[0], y_start:y_start+chunk_size[1], :]
    return chunks

def get_image_from_chunks(chunks):

    x = chunks.shape[0] * chunks.shape[2]
    y = chunks.shape[1] * chunks.shape[3]
    res = np.zeros((x,y,3))
    for i in range(x):
        chunk_x = i // chunks.shape[2]
        col = chunks[chunk_x, :, i % chunks.shape[2], :, :]
        res[i,:,:] = col.reshape((-1, 3))

    return res

def ripple(x,y):
    # (a/(1+r)) * cos((b/log(r+2))*r) where r = sqrt(x2 +y2 ), b=10, a=10
    A = 5
    B = 10

    t1 = A/(1+(math.sqrt(x**2+y**2)))
    t2 = math.cos(B/(math.log(math.sqrt(x**2+y**2)+2))*math.sqrt(x**2+y**2))
    return t1*t2

def rotate(path, angle):
    src_im = Image.open(path)
    im = src_im.rotate(angle, expand=False)
    im.save(path[:-4] + "_rotated.jpg")

# Method takes list of floats, change it to list of tuples of 3, meaning respectively R,G,B values
# then it takes those values and put them on the image and later saves it
def saveFile(out, name):
    out = out.dot(255)
    img = Image.fromarray(out.astype('uint8'), "RGB")
    img.save(name)