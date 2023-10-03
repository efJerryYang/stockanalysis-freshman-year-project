import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os


class StockYahooFinance:
    def __init__(self, stock_name: str):
        self.stock_name = stock_name
        df = pd.read_csv(os.path.join('ProjectStorage', 'StockData', self.stock_name + '.csv'))
        df.dropna(how='any', inplace=True, axis=0)  # 删去数据缺失的行
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df.drop(columns=['Date'], axis=1, inplace=True)
        proc_df = df[
            ['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]  # projection, alter
        # proc_df.to_csv(self.stock_name + '.proc', index=False)
        self.proc_dataframe = proc_df
        self.true_in = None
        self.true_out = None
        self.norm_in = {}
        self.mean_std = {}

    def get_io_array(self, days=7, rnn=True, task='Regression', select=('High',)) -> (np.ndarray, np.ndarray):
        """
        :param days: represents the days for model to gain information from in every forecast
        :param rnn: let the 'days' of inputs to be rearranged into 1 line, True enables the BPnn to work, while LSTMnn doesn't need to do this and default is True
        :param task: 'Regression' or 'Classification'
        :param select: select features to be forecast
        :return: the true_in and true_out numpy.ndarray tuple
        """
        proc_df = self.proc_dataframe
        if not rnn and task == 'Regression':
            # Designed for bp regression
            num_rows, num_cols = proc_df.shape
            # col = 9*14 + 3
            # 因为日期是不连续的，所以也要输入
            tmp_array = np.array(proc_df, dtype=np.float64)
            true_in = np.ndarray([num_rows - days, num_cols * days + 3], dtype=np.float64)
            true_out = np.ndarray([num_rows - days, len(select)], dtype=np.float64)
            for t in range(days, num_rows):
                true_in[t - days, 0:num_cols * days] = tmp_array[t - days:t].reshape(1, -1)
                true_in[t - days, num_cols * days:] = np.array(proc_df.iloc[t][['Year', 'Month', 'Day']],
                                                               dtype=np.float64)
                true_out[t - days] = np.array(proc_df.iloc[t][select], dtype=np.float64)
                # Todo:此处存疑，怀疑 .iloc[t-days]是切分错误，稍后需要检查 => 确实，否则bp会超前实际结果（相当于把答案告诉了它）
            self.true_in = true_in
            self.true_out = true_out
            return true_in, true_out
        elif not rnn and task == 'Classification':
            # Designed for bp classification
            if not len(select) == 1:
                print("Error!\n"
                      "The multi-output classification is not supported yet, "
                      "please contact the developer for version updating!")
                exit()
            num_rows, num_cols = proc_df.shape
            tmp_array = np.array(proc_df, dtype=np.float64)
            true_in = np.ndarray([num_rows - days, num_cols * days + 3], dtype=np.float64)
            true_out = np.zeros([num_rows - days], dtype=np.float64)
            for t in range(days, num_rows):
                true_in[t - days, 0:num_cols * days] = tmp_array[t - days:t].reshape(1, -1)
                true_in[t - days, num_cols * days:] = np.array(proc_df.iloc[t][['Year', 'Month', 'Day']],
                                                               dtype=np.float64)
                true_out[t - days] = np.array(
                    proc_df.iloc[t][select], dtype=np.float64
                ) - np.array(
                    proc_df.iloc[t - 1][select], dtype=np.float64)
            true_out[true_out < 0] = 0  # up: 0, 1  down: 1, 0
            true_out[true_out > 0] = 1
            true_out = np.array(true_out, dtype=np.longlong)

            # 只判断最低价，更有参考价值
            # 传统意义看最高
            self.true_in = true_in
            self.true_out = true_out
            return true_in, true_out
        elif rnn and task == 'Regression':
            # Designed for rnn
            # Todo: 样本错行1-20,21-40,...，在输出上不是那么好处理(或者说，原理上不容易想清楚)，输入直接reshape就完事了
            num_rows, num_cols = proc_df.shape
            tmp_array = np.array(proc_df, dtype=np.float64)
            true_in = np.ndarray([num_rows - days, days, num_cols], dtype=np.float64)
            # next day's date is not regarded as input
            # Todo: check whether the start and end point is are needed
            # Todo: problems here, the input is not consecutive sequence, which would result in lacking accuracy
            # This is the commonly existed problem in the entire project, so, change may not be so urgent
            for t in range(days, num_rows):
                true_in[t - days, :, :] = tmp_array[t - days:t].reshape(1, days, num_cols)
            true_out = np.array(proc_df.iloc[days:][select], dtype=np.float64)
            # Todo: intuitive method here, didn't do examine
            self.true_in = true_in
            self.true_out = true_out
            return true_in, true_out
        elif rnn and task == 'Classification':
            # Designed for rnn
            if not len(select) == 1:
                print("Error!\n"
                      "The multi-output classification is not supported yet, "
                      "please contact the developer for version updating!")
                exit()
            num_rows, num_cols = proc_df.shape
            tmp_array = np.array(proc_df, dtype=np.float64)
            true_in = np.ndarray([num_rows - days, days, num_cols], dtype=np.float64)
            # next day's date is not regarded as input
            true_out = np.zeros([num_rows - days], dtype=np.float64)
            for t in range(days, num_rows):
                true_in[t - days, :, :] = tmp_array[t - days:t].reshape(1, days, num_cols)
                true_out[t - days] = np.array(
                    proc_df.iloc[t][select], dtype=np.float64
                ) - np.array(
                    proc_df.iloc[t - 1][select], dtype=np.float64)
            true_out[true_out < 0] = 0  # up: 0, 1  down: 1, 0
            true_out[true_out > 0] = 1
            true_out = np.array(true_out, dtype=np.longlong)
            # print(true_out.sum() / len(true_out))
            # 0.5800826067003213
            # 0.5994
            # exit()
            self.true_in = true_in
            self.true_out = true_out
            return true_in, true_out
        else:
            print("\nError!\n Stop in function 'get_io_array'.")
            exit()

    def normalize(self, data_in=True, axis=1, validation=True, div_rate=0.7) -> np.ndarray:
        """
        :return: np.array, each row represents a normalized line
        """
        array = np.array(self.true_in)
        # (N, C) or (N, L, C)
        if data_in and not validation:
            plate = list(array.shape)
            plate[axis] = 1
            # (full_data_len, channel)
            # (N, C)
            mean, std = np.mean(array, axis=axis).reshape(plate), np.std(array, axis=axis).reshape(plate)
            # mean, std = np.mean(array), np.std(array)
            self.mean_std.update({'full': (mean, std)})
            self.norm_in.update({'train_valid_full': (array - mean) / std})
            return self.norm_in['train_valid_full']
        elif data_in and validation:
            # (full_data_len, seq_len, channel)
            # (N, L, C)
            train_block = int(len(array) * div_rate)
            train_x, valid_x = array[:train_block], array[train_block:]
            plate = list(train_x.shape), list(valid_x.shape)
            plate[0][axis] = 1
            plate[1][axis] = 1
            if axis >= 0:
                mean_train, std_train = np.mean(train_x, axis=axis).reshape(plate[0]), \
                                        np.std(train_x, axis=axis).reshape(plate[0])
                mean_valid, std_valid = np.mean(valid_x, axis=axis).reshape(plate[1]), \
                                        np.std(valid_x, axis=axis).reshape(plate[1])
            else:
                mean_train, std_train = np.mean(train_x), np.std(train_x)
                mean_valid, std_valid = np.mean(valid_x), np.std(valid_x)
            self.mean_std.update({'train': (mean_train, std_train)})
            self.mean_std.update({'valid': (mean_valid, std_valid)})
            train_x = (train_x - mean_train) / std_train
            ratio = 0  # 0-1
            if axis >= 0:
                offset = int((len(train_x) - len(valid_x)) * ratio)
                valid_x = (valid_x - mean_train[offset:offset + len(valid_x)]) / std_train[offset:offset + len(valid_x)]
                # valid_x = (valid_x - mean_valid) / std_valid  # 这里是用哪一部分数据来对验证集做 normalization ，需要手动操作
            else:
                valid_x = (valid_x - mean_train) / std_train
                # valid_x = (valid_x - mean_valid) / std_valid  # 这里是用哪一部分数据来对验证集做 normalization ，需要手动操作
            self.norm_in.update({'train_valid_separate': np.concatenate((train_x, valid_x), axis=0)})  # 只能沿着第0维度进行拼接
            return self.norm_in['train_valid_separate']


class StockDataset(Dataset):
    def __init__(self, data_in, data_out, transform=None, target_transform=None, task='Regression'):
        if task == 'Regression':
            self.data_out = torch.from_numpy(data_out).float()
            self.data_in = torch.from_numpy(data_in).float()
        else:
            self.data_out = torch.from_numpy(data_out)
            self.data_in = torch.from_numpy(data_in).float()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, index):
        data_in = self.data_in[index]
        data_out = self.data_out[index]
        if self.transform:
            data_in = self.transform(data_in)
        if self.target_transform:
            data_out = self.target_transform(data_out)
        return data_in, data_out
