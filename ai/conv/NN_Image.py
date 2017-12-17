from PIL import Image
import numpy as np

class NN_Image:
    def __init__(self, path):
        self.path = path
        self.org_size = (0, 0)
        self.org_mode = ''
        self.img = []

    # Method reads rgb values of resize image, divides them by 255 so they're all floats between 0 and 1.
    # Then it concatenates them into single array of floats which later will be passed to neural network
    def getNumPyArr(self):
        img = Image.open(self.path)
        self.org_mode = img.mode
        self.org_size = img.size
        #resize_img = self.resize(img, size=(32,32))
        img.load()
        pix = np.asarray(img, dtype='float64')
        normalized_pix = pix / 255
        self.img = normalized_pix
        return normalized_pix


    # Method resize given image to width of 32 and respective height
    def resize(self, img):
        width_size = 32
        resize_ratio = (width_size / float(img.size[0]))
        height_size = int((float(img.size[1]) * float(resize_ratio)))
        return img.resize((width_size, height_size), Image.ANTIALIAS)


    # Method takes list of floats, change it to list of tuples of 3, meaning respectively R,G,B values
    # then it takes those values and put them on the image and later saves it
    def saveFile(self, out, name):
        out = out.dot(255)
        img = Image.fromarray(out.astype('uint8'), self.org_mode)
        img.save(name)

    def get_image_chunks(self, chunk_size=(255, 255)):
        y_range = self.org_size[0] // chunk_size[0]
        x_range = self.org_size[1] // chunk_size[1]
        chunks = np.zeros(shape=(x_range, y_range, chunk_size[0], chunk_size[1], 3))
        for x in range(x_range):
            for y in range(y_range):
                x_start = x * chunk_size[0]
                y_start = y * chunk_size[1]
                chunks[x, y] = self.img[x_start:x_start + chunk_size[0], y_start:y_start + chunk_size[1], :]
        return chunks