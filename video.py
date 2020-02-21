import argparse

import numpy as np
import cv2

import torch


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
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EDR Video')
    parser.add_argument('-s', '--source', required=True, type=arg2source)
    parser.add_argument('-w', '--width', type=int, default=640)
    parser.add_argument('-g', '--gamma', type=arg2gamma, default='0.4, 0.67')
    args = parser.parse_args()
    print(args)

    cap = cv2.VideoCapture(args.source)

    while True:
        ret, frame = cap.read()

        if frame.shape[1] != args.width:
            frame = image_resize(frame, args.width)

        images = [frame.copy()]
        for g in args.gamma:
            tmp_img = adjust_gamma(frame.copy(), g)
            tmp_img = cv2.blur(tmp_img, (3, 3))

            images.append(tmp_img)

        merge_mertens = cv2.createMergeMertens()
        edr_raw = merge_mertens.process(images)

        edr_raw = np.clip(edr_raw * 255, 0, 255)
        edr = edr_raw.astype('uint8')

        cv2.imshow('LDR', frame)
        cv2.imshow('EDR', edr)

        if cv2.waitKey(1) == 113:
            cap.release()
            cv2.destroyAllWindows()
