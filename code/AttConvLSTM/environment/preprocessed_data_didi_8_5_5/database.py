from environment.preprocessed_data_didi_8_5_5.env_conf import *
import h5py
import numpy as np
import pickle
import time
import os
from copy import copy
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns

from scipy import stats


class MinMaxNormalization_01(object):
    '''MinMax Normalization --> [0, 1]
       x = (x - min) / (max - min).
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self._max - self._min) + self._min
        return X


def string2timestamp(strings, T=48):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) - 1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot),
                                                minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version
        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                temp = [self.pd_timestamps[i] - j * offset_frame for j in depend]
                Flag = self.check_it(temp)

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(x_c)
            if len_period > 0:
                XP.append(x_p)
            if len_trend > 0:
                XT.append(x_t)
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


class Database_Preprocess(object):
    def __init__(self):
        self.seq_data_train = None
        if ENV_CONF['READ_CACHE']:
            self.read_cache()
            print(ENV_CONF['H5_PATH'])
        else:
            self.load_data(T=48, nb_flow=2, len_closeness=ENV_CONF['CLOSENESS'], len_period=ENV_CONF['PERIOD'],
                           len_trend=ENV_CONF['TREND'], len_test=48 * 7 * 4)
            print("Over!!!")
            return

        if self.mmn is None:
            self.scale = 1292.0
        else:
            self.scale = self.mmn._max
        self.test_point = 0
        self.t1 = 0

    def read_data(self, data_path):
        self.data_path = data_path
        self.data_h5 = h5py.File(self.data_path, 'r')
        self.data_len = self.data_h5['data'].shape[0]

        seq_data = np.zeros(shape=[self.data_len, 32, 32, 2], dtype=np.float32)
        seq_data[:] = np.transpose(self.data_h5['data'][:], [0, 2, 3, 1])
        if self.seq_data_train is None:
            self.seq_data_train = seq_data
        else:
            self.seq_data_train = np.concatenate([self.seq_data_train, seq_data], axis=0)

    def get_batch(self, batch_num, seq_length=None):

        indices = np.random.choice(len(self.seq_data_train), size=batch_num)
        closeness_data = self.seq_data_train[indices, :seq_length - 2, :, :, :]
        period_data = np.expand_dims(self.seq_data_train[indices, 9], axis=1)
        trend_data = np.expand_dims(self.seq_data_train[indices, 14], axis=1)
        data = np.concatenate([trend_data, period_data, closeness_data], axis=1)
        return data

    def get_test(self, seq_length=None, is_test_data=False):
        indices = np.random.choice(len(self.seq_data_test), size=1)
        return self.seq_data_test[indices, :, :, :, :]

    def is_test_over(self):
        if self.test_point >= len(self.seq_data_test):
            self.test_point = 0
            return True
        else:
            return False

    def get_test_one_by_one1(self, seq_length):

        self.test_point += 1
        closeness_data = np.expand_dims(self.seq_data_test[self.test_point - 1, :seq_length - 2, :, :, :], axis=0)
        period_data = np.expand_dims(np.expand_dims(self.seq_data_test[self.test_point - 1, 9], axis=0), axis=1)
        trend_data = np.expand_dims(np.expand_dims(self.seq_data_test[self.test_point - 1, 14], axis=0), axis=1)
        data = np.concatenate([trend_data, period_data, closeness_data], axis=1)
        return data



    def get_test_batch_haha(self, test_batch=1024):

        rt = self.seq_data_test[self.test_point:self.test_point + test_batch, :, :, :, :]
        if len(rt.shape) < 5:
            rt = np.expand_dims(rt, axis=0)
        self.test_point += test_batch
        return rt

    def cal_rmse(self, seq_start, gt, res):
        gt = gt[:, :, :, :, :] * self.scale
        res = np.array(res[0]) * self.scale
        z = len(gt[:, seq_start + 1:, :, :, :]) * 32 * 32 * 2
        rmse = np.sqrt(np.mean(np.square(gt[:, seq_start:, :, :, :] - res[:, seq_start - 1:seq_start, :, :, :])))
        return rmse

    def load_data(self, T, nb_flow, len_closeness, len_period, len_trend,
                  len_test, preprocess_name=ENV_CONF['PREPROCESS_PKL_PATH'],
                  ):
        """
        """
        assert (len_closeness + len_period + len_trend > 0)
        # load data
        # 13 - 16
        data_all = []
        timestamps_all = list()
        for year in range(13, 17):
            fname = os.path.join(
                ENV_CONF['DATA_PATH'], 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
            print("file name: ", fname)
            self.stat(fname)
            data, timestamps = self.load_stdata(fname)
            # print(timestamps)
            # remove a certain day which does not have 48 timestamps
            data, timestamps = self.remove_incomplete_days(data, timestamps, T)
            data = data[:, :nb_flow]
            data[data < 0] = 0.
            data_all.append(data)
            timestamps_all.append(timestamps)
            print("\n")

        # minmax_scale
        data_train = np.vstack(copy(data_all))[:-len_test]
        print('train_data shape: ', data_train.shape)
        mmn = MinMaxNormalization_01()
        mmn.fit(data_train)
        data_all_mmn = [mmn.transform(d) for d in data_all]

        fpkl = open(preprocess_name, 'wb')
        for obj in [mmn]:
            pickle.dump(obj, fpkl)
        fpkl.close()

        XC, XP, XT = [], [], []
        Y = []
        timestamps_Y = []
        for data, timestamps in zip(data_all_mmn, timestamps_all):
            # instance-based dataset --> sequences with format as (X, Y) where X is
            # a sequence of images and Y is an image.
            st = STMatrix(data, timestamps, T, CheckComplete=False)
            _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
                len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
            XC.append(_XC)
            XP.append(_XP)
            XT.append(_XT)
            Y.append(_Y)
            timestamps_Y += _timestamps_Y

        XC = np.vstack(XC)
        XP = np.vstack(XP)
        XT = np.vstack(XT)
        Y = np.vstack(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
              "XT shape: ", XT.shape, "Y shape:", Y.shape)

        X = []

        for l, X_ in zip([len_closeness, len_period, len_trend], [XC, XP, XT]):
            if l > 0:
                X.append(X_)
        # for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        #     if l > 0:
        #         X_test.append(X_)
        # print('train shape:', XC_train.shape, Y_train.shape,
        #       'test shape: ', XC_test.shape, Y_test.shape)
        seq = []
        if ENV_CONF['IS_REVERSE']:
            for i in range(len_closeness):
                seq.append(X[0][:, i, :, :, :])
            for i in range(len_period):
                seq.append(X[1][:, i, :, :, :])
            for i in range(len_trend):
                seq.append(X[2][:, i, :, :, :])
            seq.append(Y)
            print('reverse')

        elif ENV_CONF['IS_154320']:
            seq.append(X[0][:, 0, :, :, :])
            for i in range(len_trend - 1, -1, -1):
                seq.append(X[2][:, i, :, :, :])
            for i in range(len_period - 1, -1, -1):
                seq.append(X[1][:, i, :, :, :])
            seq.append(X[0][:, 2, :, :, :])
            seq.append(X[0][:, 1, :, :, :])
            seq.append(Y)
            print('154320')
        elif ENV_CONF['IS_054321']:
            seq.append(Y)
            for i in range(len_trend - 1, -1, -1):
                seq.append(X[2][:, i, :, :, :])
            for i in range(len_period - 1, -1, -1):
                seq.append(X[1][:, i, :, :, :])
            seq.append(X[0][:, 2, :, :, :])
            seq.append(X[0][:, 1, :, :, :])
            seq.append(X[0][:, 0, :, :, :])
            print('054321')
        else:
            for i in range(len_trend - 1, -1, -1):
                seq.append(X[2][:, i, :, :, :])
            for i in range(len_period - 1, -1, -1):
                seq.append(X[1][:, i, :, :, :])
            for i in range(len_closeness - 1, -1, -1):
                seq.append(X[0][:, i, :, :, :])
            seq.append(Y)

        seq = np.asarray(seq)
        seq = np.transpose(seq, [1, 0, 3, 4, 2])
        if ENV_CONF['IS_SHUFFLE']:
            np.random.shuffle(seq)
            print("!!!shuffle!!!")

        self.seq_data_train = seq[:-len_test]
        self.seq_data_test = seq[-len_test:]
        print("train", self.seq_data_train.shape, "test", self.seq_data_test.shape)

        h5 = h5py.File(ENV_CONF['H5_PATH'], 'w')
        h5.create_dataset(name='seq_data_train', data=self.seq_data_train, dtype=np.float32)
        h5.create_dataset(name='seq_data_test', data=self.seq_data_test, dtype=np.float32)
        h5.close()

    def stat(self, fname):

        with h5py.File(fname) as f:
            nb_timeslot, ts_str, te_str = self.get_nb_timeslot(f)
            nb_day = int(nb_timeslot / 48)
            mmax = f['data'].value.max()
            mmin = f['data'].value.min()
            stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
                   'data shape: %s\n' % str(f['data'].shape) + \
                   '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
                   '# of timeslots: %i\n' % int(nb_timeslot) + \
                   '# of timeslots (available): %i\n' % f['date'].shape[0] + \
                   'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
                   'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
                   '=' * 5 + 'stat' + '=' * 5
            print(stat)

    def load_stdata(self, fname):
        f = h5py.File(fname, 'r')
        data = f['data'].value
        timestamps = f['date'].value
        f.close()
        return data, timestamps

    def remove_incomplete_days(self, data, timestamps, T=48):
        # remove a certain day which has not 48 timestamps
        days = []  # available days: some day only contain some seqs
        days_incomplete = []
        i = 0
        while i < len(timestamps):
            if int(timestamps[i][8:]) != 1:
                i += 1
            elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
                days.append(timestamps[i][:8])
                i += T
            else:
                days_incomplete.append(timestamps[i][:8])
                i += 1
        print("incomplete days: ", days_incomplete)
        days = set(days)
        idx = []
        for i, t in enumerate(timestamps):
            if t[:8] in days:
                idx.append(i)

        data = data[idx]
        timestamps = [timestamps[i] for i in idx]
        return data, timestamps

    def get_nb_timeslot(self, f):
        s = f['date'][0]
        e = f['date'][-1]
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    def cal_one_rmse(self, gt, res):
        gt = gt * self.scale
        res = np.array(res) * self.scale
        return np.sqrt(np.mean(np.square(gt - res)))

    def cal_rmse_haha(self, gt, res):
        gt = gt * self.scale
        res = np.array(res) * self.scale
        return np.sqrt(np.mean(np.square(gt - res), axis=1))

    # def get_mape(self, y_true, y_pred):
    #     y_true = y_true.reshape(-1)
    #     y_pred = y_pred.reshape(-1)
    #     # print('Number of sample : %d\n' % y_true.shape[0])
    #     y_true_new = []
    #     y_pred_new = []
    #     for index in range(y_true.shape[0]):
    #         if y_true[index] * self.scale > 10 - 1e-10:
    #             y_true_new.append(y_true[index] * self.scale)
    #             y_pred_new.append(y_pred[index] * self.scale)
    #
    #     # print('Number of sample whose label beyond 10: %d\n' % len(y_true_new))
    #
    #     # res = sum(abs(2 * (y_true - y_pred) / (y_true + y_pred))) / len(y_true)
    #     # res_2 = np.sqrt(np.mean((y_true - y_pred) * (y_true - y_pred)))
    #     res_3 = sum(abs((y_true_new - y_pred_new) / (y_true_new + 10))) / len(y_true)
    #
    #     return res_3

    def cal_one_mape(self, gt, res):

        gt = gt.reshape(-1) * self.scale
        res = np.array(res).reshape(-1) * self.scale

        # gt_new = gt[gt > 10 - 1e-10]
        # res_new = res[gt > 10 - 1e-10]
        gt_new = gt[gt > 0]
        res_new = res[gt > 0]
        return np.mean((np.abs(gt_new - res_new)) / (gt_new))

    def cal_mape_haha(self, gt, res):

        gt = gt.reshape([gt.shape[0], -1]) * self.scale
        res = np.array(res).reshape([gt.shape[0], -1]) * self.scale

        # gt_new = gt[gt > 10 - 1e-10]
        # res_new = res[gt > 10 - 1e-10]
        lst = []
        for i in range(gt.shape[0]):
            gt_new = gt[i, gt[i] > 0]
            res_new = res[i, gt[i] > 0]
            tmp = np.mean((np.abs(gt_new - res_new)) / (gt_new))
            lst.append(tmp)
        return np.array(lst)

    def read_cache(self, ):
        print(ENV_CONF['H5_PATH'])
        if 'beijing' not in ENV_CONF['PREPROCESS_PKL_PATH']:
            self.mmn = pickle.load(open(ENV_CONF['PREPROCESS_PKL_PATH'], 'rb'))
        else:
            self.mmn = None
        f = h5py.File(ENV_CONF['H5_PATH'], 'r')
        print(ENV_CONF['H5_PATH'])
        print(f.keys())

        for k in f.keys():
            print(k, f[k][()].shape)
        self.seq_data_train = f['seq_data_train'][()]
        self.seq_data_test = f['seq_data_test'][()]
        # print(self.seq_data_train.shape,self.seq_data_test.shape)
        f.close()

    def cal_confidence_interval(self, x):
        x1 = np.array(x)
        # x1=x1[np.where(x1<=30 )]
        # x1=x1[np.where(x1>4)]
        # sns.distplot(x1,bins=100,kde=False, rug=True)
        mean = x1.mean()

        std = x1.std()

        # interval = stats.norm.interval(0.9, mean, std)
        interval = stats.norm.interval(0.9, mean, std)
        print('{:0.2%} of the means are in conf_int_b'
              .format(((x1 >= interval[0]) & (x1 < interval[1])).sum() / float(x1.shape[0])))
        return interval


if __name__ == '__main__':
    database = Database_Preprocess()
# max value is 500 is more suitable
