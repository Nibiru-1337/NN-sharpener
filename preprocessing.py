import glob
import os
import random
import time

from skimage.filters import gaussian

from ai.conv import ImageOperations
from ai.conv.ImageOperations import ripple, rotate
from ai.conv.NN_Image import NN_Image


def blur(npImg, chunk_size=(16, 16)):
    t = time.time()
    # chunks = get_image_chunks(npImg, chunk_size=chunk_size)
    print('get_img_chunks:{}'.format(time.time() - t))

    t = time.time()
    # for x in range(chunks.shape[0]):
    #     for y in range(chunks.shape[1]):
    sigma = ripple(random.randint(0, npImg.shape[0]), random.randint(0, npImg.shape[1]))
    # if sigma < 0:
    #     sigma = 0
    # else:
    #     sigma *= 50
    sigma = 20
    # print(sigma)
    result = gaussian(npImg, sigma=sigma, multichannel=True)
    # chunks[x, y] = np.array([sigma * 16*16])
    print('gauss on chunks:{}'.format(time.time() - t))

    t = time.time()
    # result = get_image_from_chunks(chunks)
    print('get_image_from_chunks:{}'.format(time.time() - t))
    return result


def getPathsFromDir(dirPath):
    labels = []
    for x in os.walk(dirPath):
        new_list = [dirPath + x for x in x[2]]
        labels = new_list

    return labels


def blur_image(path, name, out_name):
    read_image = NN_Image(path + name)
    npImg = read_image.getNumPyArr()
    chunk_size = (npImg.shape[0], npImg.shape[1])
    result = blur(npImg, chunk_size=chunk_size)
    new_file = path + out_name
    ImageOperations.saveFile(result, new_file)


def blur_directory(path):
    pathlist = getPathsFromDir(path)

    i = 0
    while 1:
        read_image = NN_Image(path + str(i) + ".jpg")
        npImg = read_image.getNumPyArr()
        chunk_size = (npImg.shape[0], npImg.shape[1])
        result = blur(npImg, chunk_size=chunk_size)
        new_file = '.\\data\\blurred_faces_2\\' + str(i) + '.jpg'
        ImageOperations.saveFile(result, new_file)
        print('DONE:{}'.format(new_file))
        i = i + 1


def clear_blur(path):
    for pathAndFilename in glob.iglob(os.path.join(path, r'*_blur.jpg')):
        print('RM:{}'.format(pathAndFilename))
        os.remove(pathAndFilename)


def rotate_directory(path):
    for pathAndFilename in glob.iglob(os.path.join(path, r'*.jpg')):
        rotate(pathAndFilename, random.randint(0, 360))
        print('DONE:{}'.format(pathAndFilename))


def clear_rotate(path):
    for pathAndFilename in glob.iglob(os.path.join(path, r'*_rotated.jpg')):
        print('RM:{}'.format(pathAndFilename))
        os.remove(pathAndFilename)


def wat_test():
    read_image = NN_Image('.\\example.jpg')
    npImg = read_image.getNumPyArr()
    result = blur(npImg, chunk_size=(npImg.shape[0] // 50, npImg.shape[1] // 50))
    ImageOperations.saveFile(result, 'example_blur.jpg')

    rotate('.\\example.jpg', 15)
    read_image = NN_Image('.\\example_rotated.jpg')
    npImg = read_image.getNumPyArr()
    result = blur(npImg, chunk_size=(npImg.shape[0] // 50, npImg.shape[1] // 50))
    ImageOperations.saveFile(result, 'example_rotated_blur.jpg')


def main():
    directory = '.\\data\\to_blur\\'
    # clear_blur(directory)
    # clear_rotate(directory)
    # rotate_directory(directory)
    # blur_directory(directory)
    blur_image(directory,"images.jpg","blurred.jpg")
    # test()


# PROJECT 2
# IMAGE DEBLURRING
# NVIDIA RAY TRACING
# https://www.youtube.com/watch?v-bN
# https://www.youtube.com/watch?v=6eBpjEdgSm0
# capillar wave function for sigma gauss parameter
if __name__ == '__main__':
    main()
