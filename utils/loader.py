import json
import random
import torch
import numpy as np
import codecs
import copy
import pdb
import pickle
import pandas as pd
import math


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, filename, batch_size, opt, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.filename = filename
        # ************* item_id *****************
        opt["source_item_num"] = self.read_item("./dataset/" + filename + "/Alist.txt")  # itemA_number
        opt["target_item_num"] = self.read_item("./dataset/" + filename + "/Blist.txt")  # itemB_number

        # ************* sequential data *****************

        source_train_data = "./dataset/" + filename + "/traindata_new.txt"
        source_valid_data = "./dataset/" + filename + "/validdata_new.txt"
        source_test_data = "./dataset/" + filename + "/testdata_new.txt"

        train_data_file = "./dataset/" + filename + "/processed_train_data.pkl"
        val_data_file = "./dataset/" + filename + "/processed_val_data.pkl"
        test_data_file = "./dataset/" + filename + "/processed_test_data.pkl"

        self.timestamp_file = "./dataset/" + filename + "/combined_timestamp_map.csv"

        df = pd.read_csv(self.timestamp_file, header=None, delimiter=',')
        self.opt["time_num"] = len(df)

        if evaluation < 0:
            self.train_data = self.read_train_data(source_train_data)  # 返回全部交互序列的list
            data = self.preprocess()
            with open(train_data_file, "wb") as file:
                pickle.dump(data, file)
        elif evaluation == 2:
            self.test_data = self.read_test_data(source_valid_data)
            data = self.preprocess_for_predict()
            with open(val_data_file, "wb") as file:
                pickle.dump(data, file)
        else:
            self.test_data = self.read_test_data(source_test_data)
            data = self.preprocess_for_predict()
            with open(test_data_file, "wb") as file:
                pickle.dump(data, file)

        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data) % batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data) // batch_size) * batch_size]
        else:
            batch_size = 2048
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data  # 134, 256, 23, 15

    def read_item(self, fname):
        item_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                item_number += 1
        return item_number

    def timestamp_to_index(self, file):
        df = pd.read_csv(file, header=None, delimiter=',')
        timestamp_to_index = dict(zip(df[1], df[0]))
        return timestamp_to_index

    def load_buckets(self, path):
        buckets = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                buckets.append(int(line.strip()))
        return buckets

    def read_train_data(self, train_file):
        def takeSecond(elem):
            return elem[1]

        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            train_data_time = []
            for id, line in enumerate(infile):
                res = []

                line = line.strip().split("\t")[2:]
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))
                res.sort(key=takeSecond)
                res_2 = []
                res_3 = []
                for r in res:
                    res_2.append(r[0])
                    res_3.append(r[1])
                train_data.append(res_2)
                train_data_time.append(res_3)
            train_data_new = [train_data, train_data_time]

        return train_data_new

    def read_test_data(self, test_file):
        def takeSecond(elem):
            return elem[1]

        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            test_data_time = []
            for id, line in enumerate(infile):
                res = []
                line = line.strip().split("\t")[2:]
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))

                res.sort(key=takeSecond)

                res_2 = []
                res_3 = []
                for r in res:
                    res_2.append(r[0])
                    res_3.append(r[1])

                test_data.append(res_2)
                test_data_time.append(res_3)

            test_data_new = [test_data, test_data_time]
        return test_data_new

    def preprocess_for_predict(self):
        time_to_index = self.timestamp_to_index(self.timestamp_file)

        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            max_len = 15
            self.opt["maxlen"] = 15

        if self.eval == 2:
            semantic_path = "./dataset/" + self.filename + "/val_semantic_prompt.npz"
        else:
            semantic_path = "./dataset/" + self.filename + "/test_semantic_prompt.npz"
        sem_npz = np.load(semantic_path)
        m_semantic = sem_npz["emb_mixed"].astype("float32")  # [N, K]
        x_semantic = sem_npz["emb_a"].astype("float32")  # [N, K]
        y_semantic = sem_npz["emb_b"].astype("float32")  # [N, K]

        max_interval = 0
        processed = []
        for tmp in range(len(self.test_data[0])):  # the pad is needed! but to be careful.
            d = self.test_data[0][tmp]
            time = self.test_data[1][tmp]

            m_s = m_semantic[tmp]
            x_s = x_semantic[tmp]
            y_s = y_semantic[tmp]

            xd = []
            x_time = []
            x_interval_time = []
            xcnt = 1
            x_position = []
            x_ground = []

            yd = []
            y_time = []
            y_interval_time = []
            ycnt = 1
            y_position = []
            y_ground = []

            for i in range(len(d)):
                w = d[i]
                t = time[i]
                if w < self.opt["source_item_num"]:
                    xd.append(w)
                    x_time.append(time_to_index[t])
                    x_interval_time.append(t)
                    x_position.append(xcnt)
                    xcnt += 1
                else:
                    yd.append(w)
                    y_time.append(time_to_index[t])
                    y_interval_time.append(t)
                    y_position.append(ycnt)
                    ycnt += 1

            raw_time = time.copy()
            for t in range(len(time)):
                time[t] = time_to_index[time[t]]

            for i in range(len(xd) - 1, -1, -1):
                if xd[i] != self.opt["source_item_num"] + self.opt["target_item_num"]:
                    x_ground.append(xd[i])
                    xd.pop(i)
                    x_time.pop(i)
                    x_interval_time.pop(i)
                    x_position.pop(i)
                    break

            for i in range(len(yd) - 1, -1, -1):
                if yd[i] != self.opt["source_item_num"] + self.opt["target_item_num"]:
                    y_ground.append(yd[i] - self.opt["source_item_num"])
                    yd.pop(i)
                    y_time.pop(i)
                    y_interval_time.pop(i)
                    y_position.pop(i)
                    break

            if x_ground:
                value_to_remove = x_ground[0]
                index_to_remove = len(d) - 1 - d[::-1].index(value_to_remove)
                d.pop(index_to_remove)
                time.pop(index_to_remove)
                raw_time.pop(index_to_remove)

            if y_ground:
                value_to_remove = y_ground[0] + self.opt["source_item_num"]
                index_to_remove = len(d) - 1 - d[::-1].index(value_to_remove)
                d.pop(index_to_remove)
                time.pop(index_to_remove)
                raw_time.pop(index_to_remove)

            position = list(range(len(d) + 1))[1:]

            x_ground_mask_share = [0] * len(d)
            y_ground_mask_share = [0] * len(d)
            for i in range(len(d)):
                if d[i] < self.opt["source_item_num"]:
                    x_ground_mask_share[i] = 1
                else:
                    if d[i] != self.opt["source_item_num"] + self.opt["target_item_num"]:
                        y_ground_mask_share[i] = 1

            x_interval = []
            prev = None
            for num in x_interval_time:
                if num == 0:
                    x_interval.append(-1)
                else:
                    if prev is not None:
                        x_interval.append(num - prev)
                    prev = num
            if len(x_interval) == 0:
                x_interval.append(-1)
            max_interval = max(max_interval, max(x_interval))

            y_interval = []
            prev = None
            for num in y_interval_time:
                if num == 0:
                    y_interval.append(-1)
                else:
                    if prev is not None:
                        y_interval.append(num - prev)
                    prev = num
            if len(y_interval) == 0:
                y_interval.append(-1)
            max_interval = max(max_interval, max(y_interval))

            interval = []
            prev = None
            for num in raw_time:
                if num == 0:
                    interval.append(-1)
                else:
                    if prev is not None:
                        interval.append(num - prev)
                    prev = num
            max_interval = max(max_interval, max(interval))

            share_x_mask = []
            share_y_mask = []
            for i in d:
                if i < self.opt["source_item_num"]:
                    share_x_mask.append(1)
                    share_y_mask.append(0)
                elif i != self.opt["source_item_num"] + self.opt["target_item_num"]:
                    share_x_mask.append(0)
                    share_y_mask.append(1)
                else:
                    share_x_mask.append(0)
                    share_y_mask.append(0)

            if len(d) < max_len:
                position = [0] * (max_len - len(position)) + position
                x_position = [0] * (max_len - len(x_position)) + x_position
                y_position = [0] * (max_len - len(y_position)) + y_position

                x_ground_mask_share = [0] * (max_len - len(x_ground_mask_share)) + x_ground_mask_share
                y_ground_mask_share = [0] * (max_len - len(y_ground_mask_share)) + y_ground_mask_share
                share_x_mask = [0] * (max_len - len(share_x_mask)) + share_x_mask
                share_y_mask = [0] * (max_len - len(share_y_mask)) + share_y_mask

                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(xd)) + xd
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(yd)) + yd
                seq = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + d

                x_time = [0] * (max_len - len(x_time)) + x_time
                y_time = [0] * (max_len - len(y_time)) + y_time
                time = [0] * (max_len - len(time)) + time

                x_interval = [-1] * (max_len - len(x_interval)) + x_interval
                y_interval = [-1] * (max_len - len(y_interval)) + y_interval
                interval = [-1] * (max_len - len(interval)) + interval

            x_last = -1
            for id in range(len(x_position)):
                id += 1
                if x_position[-id]:
                    x_last = -id
                    break

            y_last = -1
            for id in range(len(y_position)):
                id += 1
                if y_position[-id]:
                    y_last = -id
                    break

            negative_sample_y = []
            negative_sample_x = []
            for i in range(999):
                while True:
                    sample = random.randint(0, self.opt["target_item_num"] - 1)
                    if sample != y_ground[0]:
                        negative_sample_y.append(sample)
                        break
            for i in range(999):
                while True:
                    sample = random.randint(0, self.opt["source_item_num"] - 1)
                    if sample != x_ground[0]:
                        negative_sample_x.append(sample)
                        break

            processed.append([seq, xd, yd, position, x_position, y_position, x_last, y_last, x_ground,
                              y_ground, negative_sample_x, negative_sample_y, time, x_time, y_time,
                              x_interval, y_interval, interval, m_s, x_s, y_s, x_ground_mask_share, y_ground_mask_share, share_x_mask, share_y_mask])
        max_interval = torch.log2(torch.tensor(max_interval) + 1).item()
        max_interval = max(max_interval, self.opt["max_interval"])
        self.opt["max_interval"] = max_interval
        return processed

    def preprocess(self):

        time_to_index = self.timestamp_to_index(self.timestamp_file)

        def myprint(a):
            for i in a:
                print("%6d" % i, end="")
            print("")

        """ Preprocess the data and convert to ids. """
        processed = []

        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            max_len = 15
            self.opt["maxlen"] = 15

        semantic_path = "./dataset/" + self.filename + "/train_semantic_prompt.npz"
        semantic_small_path = "./dataset/" + self.filename + "/train_small_semantic_prompt.npz"
        semantic_big_path = "./dataset/" + self.filename + "/train_big_semantic_prompt.npz"
        sem_npz = np.load(semantic_path)
        m_semantic = sem_npz["emb_mixed"].astype("float32")  # [N, K]
        x_semantic = sem_npz["emb_a"].astype("float32")  # [N, K]
        y_semantic = sem_npz["emb_b"].astype("float32")  # [N, K]
        sem_small_npz = np.load(semantic_small_path)
        m_small_semantic = sem_small_npz["emb_mixed"].astype("float32")  # [N, K]
        x_small_semantic = sem_small_npz["emb_a"].astype("float32")  # [N, K]
        y_small_semantic = sem_small_npz["emb_b"].astype("float32")  # [N, K]
        sem_big_npz = np.load(semantic_big_path)
        m_big_semantic = sem_big_npz["emb_mixed"].astype("float32")  # [N, K]
        x_big_semantic = sem_big_npz["emb_a"].astype("float32")  # [N, K]
        y_big_semantic = sem_big_npz["emb_b"].astype("float32")  # [N, K]

        max_interval = 0
        for tmp in range(len(self.train_data[0])):

            d = self.train_data[0][tmp]
            time = self.train_data[1][tmp]
            m_s = m_semantic[tmp]
            x_s = x_semantic[tmp]
            y_s = y_semantic[tmp]
            m_s_s = m_small_semantic[tmp]
            x_s_s = x_small_semantic[tmp]
            y_s_s = y_small_semantic[tmp]
            m_b_s = m_big_semantic[tmp]
            x_b_s = x_big_semantic[tmp]
            y_b_s = y_big_semantic[tmp]

            ground = copy.deepcopy(d)[1:]

            position = list(range(len(d) + 1))[1:]

            xd = []
            x_time = []
            x_interval_time = []
            xcnt = 1
            x_position = []

            yd = []
            y_time = []
            y_interval_time = []
            ycnt = 1
            y_position = []

            for i in range(len(d)):
                w = d[i]
                t = time[i]
                if w < self.opt["source_item_num"]:
                    xd.append(w)
                    x_time.append(time_to_index[t])
                    x_interval_time.append(t)
                    x_position.append(xcnt)
                    xcnt += 1
                else:
                    yd.append(w)
                    y_time.append(time_to_index[t])
                    y_interval_time.append(t)
                    y_position.append(ycnt)
                    ycnt += 1

            raw_time = time.copy()
            for t in range(len(time)):
                time[t] = time_to_index[time[t]]

            x_ground = xd[1:]
            x_ground_mask = [1] * len(x_ground)
            xd = xd[:-1]
            x_position = x_position[:-1]
            x_time = x_time[:-1]
            x_interval_time = x_interval_time[:-1]
            if len(x_ground) == 0:
                print("pass sequence x")
                continue

            y_ground = yd[1:]
            y_ground = [x - self.opt["source_item_num"] for x in y_ground]
            y_ground_mask = [1] * len(y_ground)
            yd = yd[:-1]
            y_position = y_position[:-1]
            y_time = y_time[:-1]
            y_interval_time = y_interval_time[:-1]
            if len(y_ground) == 0:
                print("pass sequence y")
                continue

            d = d[:-1]
            time = time[:-1]
            raw_time = raw_time[:-1]
            if ground[-1] < self.opt["source_item_num"]:
                for i in range(len(d) - 1, -1, -1):
                    if d[i] >= self.opt["source_item_num"]:
                        d[i] = self.opt["source_item_num"] + self.opt["target_item_num"]
                        time[i] = 0
                        raw_time[i] = 0
                        position = [0 if j == i else (x - 1 if j > i else x)
                                    for j, x in enumerate(position)]
                        break
            else:
                for i in range(len(d) - 1, -1, -1):
                    if d[i] < self.opt["source_item_num"]:
                        d[i] = self.opt["source_item_num"] + self.opt["target_item_num"]
                        time[i] = 0
                        raw_time[i] = 0
                        position = [0 if j == i else (x - 1 if j > i else x)
                                    for j, x in enumerate(position)]
                        break

            x_ground_mask_share = [0] * len(d)
            y_ground_mask_share = [0] * len(d)
            for i in range(len(d)):
                if d[i] < self.opt["source_item_num"]:
                    x_ground_mask_share[i] = 1
                else:
                    if d[i] != self.opt["source_item_num"] + self.opt["target_item_num"]:
                        y_ground_mask_share[i] = 1

            x_interval = []
            prev = None
            for num in x_interval_time:
                if num == 0:
                    x_interval.append(-1)
                else:
                    if prev is not None:
                        x_interval.append(num - prev)
                    prev = num
            if len(x_interval) == 0:
                x_interval.append(-1)
            max_interval = max(max_interval, max(x_interval))

            y_interval = []
            prev = None
            for num in y_interval_time:
                if num == 0:
                    y_interval.append(-1)
                else:
                    if prev is not None:
                        y_interval.append(num - prev)
                    prev = num
            if len(y_interval) == 0:
                y_interval.append(-1)
            max_interval = max(max_interval, max(y_interval))

            interval = []
            prev = None
            for num in raw_time:
                if num == 0:
                    interval.append(-1)
                else:
                    if prev is not None:
                        interval.append(num - prev)
                    prev = num
            max_interval = max(max_interval, max(interval))

            share_x_mask = []
            share_y_mask = []
            for i in d:
                if i < self.opt["source_item_num"]:
                    share_x_mask.append(1)
                    share_y_mask.append(0)
                elif i != self.opt["source_item_num"] + self.opt["target_item_num"]:
                    share_x_mask.append(0)
                    share_y_mask.append(1)
                else:
                    share_x_mask.append(0)
                    share_y_mask.append(0)


            if len(d) < max_len:
                position = [0] * (max_len - len(position)) + position
                x_position = [0] * (max_len - len(x_position)) + x_position
                y_position = [0] * (max_len - len(y_position)) + y_position

                x_ground = [self.opt["source_item_num"]] * (max_len - len(x_ground)) + x_ground
                y_ground = [self.opt["target_item_num"]] * (max_len - len(y_ground)) + y_ground

                x_ground_mask = [0] * (max_len - len(x_ground_mask)) + x_ground_mask
                y_ground_mask = [0] * (max_len - len(y_ground_mask)) + y_ground_mask
                x_ground_mask_share = [0] * (max_len - len(x_ground_mask_share)) + x_ground_mask_share
                y_ground_mask_share = [0] * (max_len - len(y_ground_mask_share)) + y_ground_mask_share
                share_x_mask = [0] * (max_len - len(share_x_mask)) + share_x_mask
                share_y_mask = [0] * (max_len - len(share_y_mask)) + share_y_mask

                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(xd)) + xd
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(yd)) + yd
                x_time = [0] * (max_len - len(x_time)) + x_time
                y_time = [0] * (max_len - len(y_time)) + y_time
                time = [0] * (max_len - len(time)) + time

                x_interval = [-1] * (max_len - len(x_interval)) + x_interval
                y_interval = [-1] * (max_len - len(y_interval)) + y_interval
                interval = [-1] * (max_len - len(interval)) + interval

                d = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + d
            else:
                print("pass")

            processed.append(
                [d, xd, yd, position, x_position, y_position, x_ground, y_ground, x_ground_mask, y_ground_mask,
                 time, x_time, y_time, x_interval, y_interval, interval, m_s, x_s, y_s, m_s_s, x_s_s, y_s_s,
                 m_b_s, x_b_s, y_b_s, share_x_mask, share_y_mask,
                 x_ground_mask_share, y_ground_mask_share])
        max_interval = torch.log2(torch.tensor(max_interval) + 1).item()
        self.opt["max_interval"] = max_interval
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval != -1:
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]),
                    torch.LongTensor(batch[3]), torch.LongTensor(batch[4]), torch.LongTensor(batch[5]),
                    torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]),
                    torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]),
                    torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]),
                    torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]),
                    torch.FloatTensor(np.array(batch[18])), torch.FloatTensor(np.array(batch[19])),
                    torch.FloatTensor(np.array(batch[20])), torch.LongTensor(batch[21]), torch.LongTensor(batch[22]),
                    torch.LongTensor(batch[23]), torch.LongTensor(batch[24]))
        else:
            batch = list(zip(*batch))

            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]),
                    torch.LongTensor(batch[3]), torch.LongTensor(batch[4]), torch.LongTensor(batch[5]),
                    torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]),
                    torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]),
                    torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]),
                    torch.LongTensor(batch[15]), torch.FloatTensor(np.array(batch[16])), torch.FloatTensor(np.array(batch[17])),
                    torch.FloatTensor(np.array(batch[18])), torch.FloatTensor(np.array(batch[19])), torch.FloatTensor(np.array(batch[20])),
                    torch.FloatTensor(np.array(batch[21])), torch.FloatTensor(np.array(batch[22])), torch.FloatTensor(np.array(batch[23])),
                    torch.FloatTensor(np.array(batch[24])), torch.LongTensor(batch[25]), torch.LongTensor(batch[26]),
                    torch.LongTensor(batch[27]), torch.LongTensor(batch[28]))

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)