# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
import pickle as pkl

from utils.augment import DataAug


class MathFormulaDataset(object):
    """
    印刷和手写混合数学公式数据集

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        config,
        feature_file: str = "offline-train.pkl",
        label_file: str = "train_caption.txt",
        dictionary_file: str = "dictionary.txt",
    ) -> None:
        super(MathFormulaDataset, self).__init__()
        self.config = config

        # 数据集
        fp = open(feature_file, 'rb')
        self.features = pkl.load(fp)
        fp.close()

        # 标签
        fp2 = open(label_file, 'r', encoding='utf-8')
        self.labels = fp2.readlines()
        fp2.close()

        # 字典
        self.dictionary = self.load_dict(dictionary_file)

        # label:str -> int
        self.targets = {}
        for l in self.labels:
            tmp = l.strip().split()
            uid = tmp[0]
            w_list = []
            for w in tmp[1:]:
                if self.dictionary.__contains__(w):
                    w_list.append(self.dictionary[w])
                else:
                    print('a word not in the dictionary !! sentence ', uid,
                          'word ', w)
                    sys.exit()
            self.targets[uid] = w_list

        print("Finish create targets...")
        print("Target size is : ", len(self.targets))

        self.image_size = {}
        for uid, fea in self.features.items():
            self.image_size[uid] = fea.shape[1] * fea.shape[2]
        # sorted by sentence length, return a list with each triple element
        self.image_size = sorted(self.image_size.items(),
                                 key=lambda d: d[1],
                                 reverse=True)

    def load_dict(self, dictFile: str):
        """
        load dictionary
        
        Args:
            dictFile (str): _description_

        Returns:
            _type_: _description_
        """
        fp = open(dictFile, encoding='utf-8')
        stuff = fp.readlines()
        fp.close()
        lexicon = {}
        for l in stuff:
            w = l.strip().split()
            lexicon[w[0]] = int(w[1])
        print('total Latex class: ', len(lexicon))
        return lexicon

    def data_iterator(self):
        """
        数据集

        Returns:
            _type_: _description_
        """
        feature_batch = []
        label_batch = []
        feature_total = []
        label_total = []
        uidList = []
        biggest_image_size = 0

        i = 0
        for uid, size in self.image_size:
            if size > biggest_image_size:
                biggest_image_size = size
            fea = self.features[uid]
            lab = self.targets[uid]
            batch_image_size = biggest_image_size * (i + 1)
            if len(lab) > self.config.MAX_LEN:
                print('sentence', uid, 'length bigger than',
                      self.config.MAX_LEN, 'ignore')
            elif size > self.config.MAX_IMAGE_SIZE:
                print('image', uid, 'size bigger than',
                      self.config.MAX_IMAGE_SIZE, 'ignore')
            else:
                uidList.append(uid)
                if batch_image_size > self.config.BATCH_IMAGE_SIZE or i == self.config.BATCH_SIZE:  # a batch is full
                    feature_total.append(feature_batch)
                    label_total.append(label_batch)
                    i = 0
                    biggest_image_size = size
                    feature_batch = []
                    label_batch = []
                    feature_batch.append(fea)
                    label_batch.append(lab)
                    i += 1
                else:
                    feature_batch.append(fea)
                    label_batch.append(lab)
                    i += 1

        # last batch
        feature_total.append(feature_batch)
        label_total.append(label_batch)
        print('total ', len(feature_total), 'batch data loaded')
        return list(zip(feature_total, label_total)), uidList
    
    def _dynamic_ratio(self, height_list):
        """
        统一batch图片高度，计算批量数据中高度的归一化比例

        Args:
            height_list (list): batch数据集高度集合

        Returns:
            _type_: _description_
        """
        # if len < 3 all ratio is 1.0
        if len(height_list) < 3:
            return [1 for item in height_list]
        sort_height_list = sorted(height_list)
        length = len(sort_height_list)
        median_height = sort_height_list[length // 2]
        ratio_list = []
        for height in height_list:
            ratio = 1
            if height > median_height * 1.2:
                ratio = random.uniform(1, 1.2)
                ratio = ratio * median_height / height
            ratio_list.append(ratio)
        return ratio_list

    def prepare_data_bidecoder(self, images, labels):
        """
        数据预处理
        """
        heights_x = [s.shape[1] for s in images]
        ratio_list = self._dynamic_ratio(heights_x)
        
        images_x = []
        for i in range(len(ratio_list)):
            imsrc = images[i].transpose(1,2,0)
            ratio = ratio_list[i]
            if ratio != 1:
                imsrc = cv2.resize(imsrc, (0, 0),fx=ratio,fy=ratio,interpolation=cv2.INTER_CUBIC)
                if len(imsrc.shape) < 3:
                    imsrc=np.expand_dims(imsrc,axis=-1)
            images_x.append(imsrc)
            
        heights_x = [s.shape[0] for s in images_x]
        widths_x = [s.shape[1] for s in images_x]
        lengths_y = [len(s) for s in labels]
        n_samples = len(heights_x)
        max_height_x = np.max(heights_x)
        max_width_x = np.max(widths_x)
        maxlen_y = np.max(lengths_y) + 1
        
        #L2R  y_in: <sos> y1, y2, ..., yn
        #L2R  y_out: y1, y2, ..., yn, <eos>
        x = np.zeros((n_samples, self.config.INPUT_CHANNELS, max_height_x,max_width_x)).astype(np.float32)
        y_in = np.zeros((maxlen_y,n_samples)).astype(np.int64)  # <sos> must be 0 in the dict
        y_out = np.ones((maxlen_y,n_samples)).astype(np.int64)  # <eos> must be 1 in the dict

        x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
        y_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)

        for idx, [s_x, s_y] in enumerate(zip(images_x, labels)):
            # 随机偏移
            h_start = random.randint(0, max_height_x - heights_x[idx])
            w_start = random.randint(0, max_width_x - widths_x[idx])
            pastim = np.ones((max_height_x, max_width_x, 3), dtype=np.uint8) * 255
            pastim[h_start:h_start + heights_x[idx],w_start:w_start + widths_x[idx], :] = s_x
            
            # 数据增强
            augim = DataAug().aug_img(pastim)
            x[idx] = 1.0 - augim / 255.
            x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
            
            # L2R标签
            y_in[1:(lengths_y[idx] + 1), idx] = s_y
            y_out[:lengths_y[idx], idx] = s_y
            y_mask[:lengths_y[idx] + 1, idx] = 1.
        
        randi = random.randint(0, 4)
        randf = random.uniform(0.6, 1)
        if randi == 0:
            x = 1.0 - x
        elif randi == 1:
            x = randf * x
        elif randi == 2:
            x = 1.0 - x
            x = randf * x
            
        
            
            
        #R2L: y_in:  <eos> yn, yn-1, ..., y3, y2, y1
        #R2L: y_out: yn, yn-1, ..., y2, y1, <sos>
        y_reverse_in = np.ones(
            (maxlen_y,
             n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
        y_reverse_out = np.zeros(
            (maxlen_y,
             n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
        y_reverse_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)

        for idx, [s_x, s_y] in enumerate(zip(images_x, labels)):
            # R2L标签
            y_reverse_in[1:(lengths_y[idx] + 1), idx] = s_y[::-1]
            y_reverse_out[:lengths_y[idx], idx] = s_y[::-1]
            y_reverse_mask[:lengths_y[idx] + 1, idx] = 1.

        return x, x_mask, y_in, y_out, y_mask, y_reverse_in, y_reverse_out, y_reverse_mask


if __name__ == "__main__":
    import os
    import sys
    MODULE_PATH = os.path.abspath(".")
    if MODULE_PATH not in sys.path:
        sys.path.append(MODULE_PATH)

    import config

    data = MathFormulaDataset(config=config,
                              feature_file="data/offline-train.pkl",
                              label_file="data/train_caption.txt",
                              dictionary_file="data/dictionary.txt")
    data_iterator,uid_list=data.data_iterator()

    import time
    alllength=0
    for X, Y in data_iterator:
        print(len(X))
        x, x_mask, y_in, y_out, y_mask, y_reverse_in, y_reverse_out, y_reverse_mask=data.prepare_data_bidecoder(X,Y)
        alllength += len(X)
        # x, x_mask, y, y_mask = aug(X,Y)
        # print(y)
    print("batch final")
    print("all length is ", alllength)