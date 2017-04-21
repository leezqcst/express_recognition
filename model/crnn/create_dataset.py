import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)  # read from string
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)  # get img matrix
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList,
                  labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = b'image-%09d' % cnt
        labelKey = b'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if lexiconList:
            lexiconKey = b'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache[b'num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def sort_key(x):
    temp = x.split('.')
    return int(temp[0])


if __name__ == '__main__':
    root_path = '/home/sherlock/Documents/express_recognition'
    img_root_path = os.path.join(root_path,
                                 'data/train/telephone/telephone')
    label_root_path = os.path.join(root_path,
                                   'data/train/telephone/label_tele.txt')
    createdata_root_path = os.path.join(roo_path,
                                        'data/train/telephone/tele_data')

    img_list = os.listdir(img_root_path)
    img_list.sort(key=sort_key)
    img_path = []
    for i in range(len(img_list)):
        img_path.append(img_root_path + img_list[i])
    with open(label_root_path) as f:
        tele_num = f.readlines()
    label_list = [tele_num[i][-12: -1] for i in range(len(tele_num))]
    createDataset(createdata_root_path, img_path, label_list)
