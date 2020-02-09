import argparse

import cv2
import numpy as np

def arg2source(arg):
    if type(arg) is int:
        return arg
    elif type(arg) is str:
        if arg.isdigit():
            return int(arg)
        else:
            return arg

def arg2gamma(arg):
    gammas = arg.split(',')

    for i in range(len(gammas)):
        gammas[i] = float(gammas[i])

    return gammas

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    if width is None and height is None:
        return image

    (h, w) = image.shape[:2]

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EDR Image')
    parser.add_argument('-i', '--image', required=True, type=arg2source)
    parser.add_argument('-w', '--width', type=int, default=640)
    parser.add_argument('-g', '--gamma', type=arg2gamma, default='1.5,3')
    args = parser.parse_args()
    print(args)

    frame = cv2.imread(args.image)

    images = [frame.copy()]
    for g in args.gamma:
        images.append(adjust_gamma(frame.copy(), g))

    merge_mertens = cv2.createMergeMertens()
    edr_raw = merge_mertens.process(images)

    edr_raw = np.clip(edr_raw * 255, 0, 255)
    edr = edr_raw.astype('uint8')

    cv2.imshow('LDR', image_resize(frame, 1280))
    cv2.imshow('EDR', image_resize(edr, 1280))

    cv2.waitKey()
    cv2.destroyAllWindows()