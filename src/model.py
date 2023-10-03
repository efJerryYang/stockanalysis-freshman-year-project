from datetime import timedelta, datetime
import os
import numpy as np
# import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
# import dataproc
# import plotdraw
from plotdraw import show_accuracy_through_epoch, show_pred_true_through_day, show_forecast
from dataproc import StockYahooFinance, StockDataset


# loss 的计算很混乱， 统一确定后修改
class Config:
    # 基本参数，在变更股票时修改
    stock_name = '^IXIC'  # 即StockData内股票名称缩写
    task = 'Regression'  # 目前只做Regression
    select = ['Low', 'High', 'Open', 'Close', 'Adj Close']
    # ['Low', 'High', 'Open', 'Close', 'Adj Close']     # Prices
    # ['Volume']                                        # Volume
    # 目前以Prices居多，近期才发现Volume可以被学习，故逐步开始添加计算Volume的模型
    # 目前RNN均为单层结构，无法学习到Volume规律，故只能由BP处理，而且Volume复杂程度较高，可能连同BP也需要加层数
    # Todo:添加RNN层数选择

    # 模型训练参数和类别选择，每次训练完之后可能修改
    create_new = False  # 是否新建模型，如果否，需要修改save_name为加载的模型的名称
    model_type = 'BP'  # 'LSTM', 'GRU', 'BP'，改成RNN之后必须 rnn=True，否则数据处理后的数据维度报错
    rnn = False  # 是否是RNN，因数据维度不同需要专门区分

    div_rate = 0.72  # 数据集切分比例，num_train:num_valid = div_rate:(1-div_rate)
    save_model = True  # 当次训练之后是否保存模型
    save_record = True  # 当次训练之后是否保存训练记录
    # 注意，如果save_model, save_record真值不一致，会新建tmp文件，目的是为了让模型和记录匹配
    # tmp模型保存为 save_name + '.tmp'
    # tmp记录保存为 save_name + '.tmp.npz'
    # Todo:将记录保存为save_folder + ('.tmp' + )'.npz'
    # Todo:将模型保存为save_folder + ('.tmp' + )'.pth'
    # 但因不改变功能实现，所以定位为闲的没事干再改

    path = os.path.join('ProjectStorage', 'Model', task, stock_name, model_type)  # 保存的路径，目前修改
    # Todo:将保存路径重新思考再做修改，结题不考虑
    save_name = '^IXIC_BP_20211030_023237_Prices.pth'
    # '^IXIC_LSTM_20211030_012150_Prices.pth' wll trained 0.72 1e-4 8
    # 'NVAX_GRU_20211029_170327_Prices.pth'

    '''
    'NVAX_BP_20211029_105023_Prices.pth' well trained
    'BTBT_BP_20211029_010459_Prices.pth'  # 0.668 6 1e-4
    'EFA_BP_20211024_171956_Volume.pth'
    'EFA_LSTM_20211024_164650_Volume.pth'
    'EFA_BP_20211024_163732_Volume.pth'
    'EFA_BP_20211022_233825_Prices.pth'
    'FCEL_BP_20211022_232644_Prices.pth'
    'FNMAT_BP_20211022_140554_Prices.pth'
    'FNMAT_GRU_20211021_190135_Prices.pth' 暂存 0.945 1e-4, batch 4
    'FCEL_LSTM_20211021_161651_Prices.pth' bad trained, div_rate=0.28, 1e-3, 16
    'FCEL_LSTM_20211021_150747_Prices.pth' bad trained, div_rate should much smaller (0.2, 0.6)
    'EFA_LSTM_20211021_140402_Prices.pth' 0.8545 1e-3 32暂存
    'EFA_GRU_20211021_132651_Prices.pth' 0.96
    'BTBT_GRU_20211021_125937_Prices.pth' 0.8 1e-3暂存
    'BB_LSTM_20211020_230021_Prices.pth'
    'AMC_GRU_20211020_222910_Prices.pth'
    'AAPL_LSTM_20211020_193854_Prices.pth'
    '^RUT_GRU_20211020_184351_Prices.pth'
    '^RUT_GRU_20211020_184351_Prices.pth'
    '^IXIC_BP_20211020_180130_Prices.pth'
    '^IXIC_GRU_20211020_172349_Prices.pth'
    '^IXIC_LSTM_20211020_170434_Prices.pth'
    '^IXIC_BP_20211020_164622_Prices.pth'
    '^IXIC_LSTM_20211020_151249_Prices.pth'
    '^IXIC_LSTM_20211020_151249_Prices.pth'

    'NFLX_LSTM_20211019_213051_Prices.pth'
    '^RUT_BP_20211017_224656_Prices.pth'
    '^IXIC_GRU_20211017_220653_Prices.pth'
    AAPL_GRU_20211017_164618_Prices.pth
    AAPL_LSTM_20211017_163736_Prices.pth
    AAPL_BP_20211017_161743_Prices.pth
    '''
    save_folder = save_name.split('.')[0]  # 模型保存的文件夹名，文件夹内存放model、record、Plot

    # 这一部分是训练参数，基本不做修改
    validation = True  # 是否切分验证集，做训练都要切分，切分比例为div_rate
    days = 14  # 模型预测所依赖的前序天数，会影响input_size
    axis = 0  # 数据标准化维度，目前0维度对于股票数据是合理的
    data_normalize = True  # 是否做标准化，默认是
    shuffle = False  # 是否做shuffle，默认否

    # 以下为需要训练经常调的参数
    epochs = 20
    learning_rate = 1e-4
    batch_size = 32  # batch_size 目前看来对结果影响不大
    sample_training = 200  # (做完每轮训练喂数据预测结果)取样天数，建议第一轮是取接近全数据集的长度查看数据分布
    sample_evaluation = 200  # (做完每轮训练不喂数据迭代结果)取样天数，建议第一轮是取接近全数据集的长度查看数据分布
    demo_ratio = 0.7  # 取样展示的切分比例，这里就是train:valid 和train:eval
    ymargin_rate = 0.5  # 纵坐标的取值, y_scale = [min(all y) * (1 - ymargin_rate), max(all y) * (1 + ymargin_rate)]
    xmargin_day = 5  # 做evaluation的时候，横坐标最右边空余的天数，主要是为了展示直观
    fig_dpi = 800  # 和图片的清晰度有关，这里取800是参考值
    target_date = '2021-11-30'  # 预测的目标日期

    # 随机数种子，一个是numpy的，一个是torch的，都默认为0
    np_random_seed = 0
    torch_manual_seed = 0

    # 以下参数是程序运行时生成，不做初始化
    train_record = {'acc': [], 'loss': []}  # 用于在训练过程中记录一轮的准确度和损失值，最终只保存最后一轮的结果
    valid_record = {'acc': [], 'loss': []}  # 用于在运行validation set过程中记录一轮的准确度和损失值，最终只保存最后一轮的结果
    train_dict = {}  # 类似上面的字典，只是记录true和pred值
    valid_dict = {}  # 类似上面的字典，只是记录true和pred值
    record_name = None  # 临时加载记录的名字，就是模型的npz记录，运行时生成
    data = None  # 临时加载的输入数据
    model = None  # 临时加载模型
    time_postfix = None  # 临时加载时间后缀
    create_time = None  # 临时加载模型创建的时间
    loss_fn = None
    optimizer = None
    div_day = None  # 由div_rate计算得到的第div_day天
    train_dataloader = None
    valid_dataloader = None
    device = None
    least_length = len('__20211024_171956_.pth')


class Logger:
    def __init__(self):
        """
        这是记录模型训练数据的类，不必理会。
        """
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.title = ['Time', 'Operation', 'Model Name', 'Task', 'Model Type',
                      'Epochs', 'Learning Rate', 'Save Model', 'Save Record',
                      'Train Loss', 'Valid Loss', 'Train Acc', 'Valid Acc']
        self.log = []
        self.title_all = ['Stock Name', 'Model Name', 'Task', 'Model Type', 'Total Epochs', 'Time',
                          'Train Loss', 'Valid Loss', 'Train Acc', 'Valid Acc']
        self.log_dict = {'Time': current_time}

    def create_new_log(self, path, filename, detailed):
        with open(os.path.join(path, filename), 'w') as f:
            if detailed:
                f.write(','.join(self.title) + '\n')
            else:
                f.write(','.join(self.title_all) + '\n')

    def add_final_state(self, config):
        if config.create_new:
            self.log_dict.update({'Operation': 'create'})
        else:
            self.log_dict.update({'Operation': 'load'})
        self.log_dict.update({
            'Model Name': config.save_name,
            'Task': config.task,
            'Model Type': config.model_type,
            'Epochs': str(config.epochs),
            'Learning Rate': str(config.learning_rate),
            'Save Model': str(config.save_model),
            'Save Record': str(config.save_record),
        })
        self.log_dict.update({
            'Train Loss': str(np.array(config.train_record['loss']).mean()),
            'Valid Loss': str(np.array(config.valid_record['loss']).mean()),
            'Train Acc': str(np.array(config.train_record['acc']).mean()),
            'Valid Acc': str(np.array(config.valid_record['acc']).mean()),
        })
        filename = os.path.join(config.path, 'log.csv')
        if not os.path.exists(filename):
            self.create_new_log(config.path, 'log.csv', detailed=True)

        with open(os.path.join(config.path, 'log.csv'), 'a') as f:
            self.log = [self.log_dict[key] for key in self.title]
            f.write(','.join(self.log) + '\n')

    def add_training_state(self, config):
        self.log_dict.update({
            'Total Epochs': str(config.epochs),
            'Stock Name': config.stock_name,
        })
        log_path = os.path.join('ProjectStorage', 'Log')
        filename = os.path.join('ProjectStorage', 'Log', 'training.csv')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            self.create_new_log(log_path, 'training.csv', detailed=False)
        with open(filename, 'a') as f:
            self.log = [self.log_dict[key] for key in self.title_all]
            f.write(','.join(self.log) + '\n')
        # df = pd.read_csv(path)
        # print(df)
        # 这里怎么更新的还是问题，先全部不区分的写入
        # pass

    def add_trained_state(self, config):
        self.log_dict.update({
            'Total Epochs': str(config.epochs),
            'Stock Name': config.stock_name,
        })
        log_path = os.path.join('ProjectStorage', 'Log')
        filename = os.path.join('ProjectStorage', 'Log', 'trained.csv')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(filename):
            self.create_new_log(log_path, 'trained.csv', detailed=False)
        with open(filename, 'a') as f:
            self.log = [self.log_dict[key] for key in self.title_all]
            f.write(','.join(self.log) + '\n')


class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, num_layers=1):
        super(LSTM_Regression, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout=0.5,
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # (N, L, C)
        b, s, h = x.size()
        lstm_out, _ = self.lstm(x.view(s, b, h))
        s, b, h = lstm_out.size()
        linear_out = self.linear(lstm_out[-1, :, :].view(b, h))
        return linear_out


class BP_Regression(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(BP_Regression, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        logits = self.fc(x)
        return logits


class GRU_Regression(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, num_layers=1):
        super(GRU_Regression, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        b, s, h = x.size()
        gru_out, _ = self.gru(x.view(s, b, h))
        s, b, h = gru_out.size()
        linear_out = self.linear(gru_out[-1, :, :].view(b, h))
        return linear_out


def train_regression(dataloader, model, loss_fn, optimizer, device, train_dict):
    size = len(dataloader.dataset)  # full length of training set
    model.train()
    avg_acc = 0.0
    avg_loss = 0
    train_dict.update({'pred': [], 'true': []})  # 这里每次开始时清空原来存放的数据
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        # print(pred.shape,y.shape)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        y_size = 1  # 这里y_size是为了正确计算acc而获取到的y的总数据量
        for t in y.shape:
            y_size *= t
        accuracy = (1 - (np.abs((pred - y).detach()) / y.detach()).sum() / y_size).item()
        avg_acc += accuracy * y.shape[0]
        avg_loss += loss.item()
        # 这里是记录预测值和真实值
        train_dict['pred'].extend([np.array(x, dtype=np.float64) for x in pred.detach()])
        train_dict['true'].extend([np.array(x, dtype=np.float64) for x in y])
        if batch % 128 == 0:
            # 因为最后一个batch长度可能不到batch_size，这里是为了显示的准确而修改了官方文档的tutorial写法
            current = (batch - 1) * dataloader.batch_size + len(x)
            print(f"loss: {loss:>9.5f} accuracy: {accuracy * 100:>7.2f}% [{current:>5d}/{size:>5d}]")

    avg_acc /= size
    avg_loss /= size

    return avg_acc, avg_loss


def valid_regression(dataloader, model, loss_fn, device, valid_dict):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    # loss_list, acc_list = valid_dict['loss'], valid_dict['acc']
    test_loss = 0.0
    accuracy = 0.0
    valid_dict.update({'pred': [], 'true': []})
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            accuracy += ((np.abs(pred - y) / y).sum()).item()
            valid_dict['pred'].extend([np.array(x, dtype=np.float64) for x in pred.detach()])
            valid_dict['true'].extend([np.array(x, dtype=np.float64) for x in y])

    test_loss /= size
    accuracy = 1 - accuracy / size
    print(f"Test Error:\n Accuracy: {accuracy * 100:>12.2f}%, Avg loss: {test_loss:>12.5f} \n")
    return accuracy, test_loss


def preparation(config):
    # 设置随机数种子确保可复现
    np.random.seed(config.np_random_seed)
    torch.manual_seed(config.torch_manual_seed)

    data = StockYahooFinance(config.stock_name)  # 数据是从YahooFinance下载下来被处理的，这里的写法主要是为以后留修改空间
    config.data = data  # 添加一个贴在data上的标签config.data
    true_in, true_out = data.get_io_array(
        days=config.days,
        rnn=config.rnn,
        task=config.task,
        select=config.select,
    )  # 取得输入输出的np.ndarray格式数据
    config.div_day = int(len(true_in) * config.div_rate)  # 从div_day条数据开始就是validation set了

    norm_in = data.normalize(
        data_in=config.data_normalize,
        axis=config.axis,
        validation=config.validation,
        div_rate=config.div_rate,
    )  # 只对输入做标准化
    # Train, validation set division (though has done in normalize, still needed here)
    # 切分数据集，根据div_day确定切片位置
    # plotdraw.show_norm_true(config, 50)
    train_x, valid_x = norm_in[:config.div_day], norm_in[config.div_day:]
    train_y, valid_y = true_out[:config.div_day], true_out[config.div_day:]

    # 用DataSet类来封装，用DataLoader来加载数据
    # 本来是DataLoader为了方便做shuffle和normalization，但我们不做shuffle，而normalization方向的指定我没学怎么用DataSet来处理。
    # 这里主要是为了代码格式的统一，方便以后复用
    data_train = StockDataset(train_x, train_y, task=config.task)
    data_valid = StockDataset(valid_x, valid_y, task=config.task)
    config.train_dataloader = DataLoader(data_train, batch_size=config.batch_size, shuffle=config.shuffle)
    config.valid_dataloader = DataLoader(data_valid, batch_size=config.batch_size, shuffle=config.shuffle)

    # choose available device
    # 用cpu还是gpu这个没调好，不过目前也只用到cpu在跑
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    config.device = device
    if config.task == 'Regression':
        if config.model_type == 'LSTM':
            model = LSTM_Regression(
                input_size=norm_in.shape[2],  # (N,L,C)
                output_size=len(config.select),  # (N,C), 由于C=1，会导致这一维度消失
            )
            print(model)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
            )
        elif config.model_type == 'GRU':
            model = LSTM_Regression(
                input_size=norm_in.shape[2],  # (N,L,C)
                output_size=len(config.select),  # (N,C), 由于C=1，会导致这一维度消失
            )
            print(model)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
            )

        elif config.model_type == 'BP':
            model = BP_Regression(
                input_size=norm_in.shape[1],  # (N,C)
                output_size=len(config.select),
            )
            print(model)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
            )
        elif config.model_type == 'HMM':
            exit('Error!\nHMM is not available!')
        else:
            exit('Error!\nUnrecognized model_type!')

    if config.create_new:  # 创建新模型，所以要获取系统时间
        # time postfix
        time_format = '%Y%m%d_%H%M%S'  # 20211008_140844
        create_time_postfix = datetime.now().strftime(time_format)
        config.time_postfix = create_time_postfix
        # model name
        model_type = config.model_type
        if len(config.select) == 5:
            target_postfix = 'Prices'
        elif len(config.select) == 1:
            target_postfix = config.select[0]
        else:
            target_postfix = '_'.join(config.select)
        entire_model_name = '_'.join([config.stock_name, model_type, create_time_postfix, target_postfix])
        save_name = entire_model_name + '.pth'
        config.save_name = save_name
        config.save_folder = config.save_name.split('.')[0]  # save_folder就是save_name去掉.pth后缀

    elif len(config.save_name) > config.least_length:  # 一个典型的模型名字,长度大于22 '__20211024_171956_.pth'
        model.load_state_dict(torch.load(os.path.join(config.path, config.save_folder, config.save_name)))
        config.time_postfix = '_'.join(config.save_name.split('.')[0].split('_')[2:4])

    else:
        print("'save_name' undefined")
        exit()
    config.model = model.to(config.device)
    config.loss_fn = loss_fn
    config.optimizer = optimizer


def run_regression_training(config, log, draw=True):
    # 训练，for循环核心的两条语句是tmp_list被赋值的位置，tmp_list里面是函数返回的当轮(avg_acc, avg_loss)
    # 其他语句是在做记录(xxx.append(tmp_list[i]))或者展示结果(print语句)
    for t in range(config.epochs):
        print(f"Epoch {t + 1}\n-----------------------------------------")
        tmp_list = train_regression(config.train_dataloader, config.model, config.loss_fn, config.optimizer,
                                    config.device, config.train_dict)
        print(f"Train Error:\n Accuracy: {(100 * tmp_list[0]):>12.2f}%, Avg loss: {tmp_list[1]:>12.5f} \n")
        config.train_record['acc'].append(tmp_list[0]), config.train_record['loss'].append(tmp_list[1])
        tmp_list = valid_regression(config.valid_dataloader, config.model, config.loss_fn, config.device,
                                    config.valid_dict)
        config.valid_record['acc'].append(tmp_list[0]), config.valid_record['loss'].append(tmp_list[1])
    print("Done!")

    if config.save_model:  # 根据是否保存模型确定是否创建临时数据文件，即确定record_name是否为.tmp.npz
        torch.save(config.model.state_dict(), os.path.join(config.path, config.save_folder, config.save_name))
        print(f"Saved PyTorch Model State to\t\t{config.save_name}")
        config.record_name = config.save_name + '.npz'
    else:
        print("\nDid not save training data")
        config.record_name = config.save_name + '.tmp.npz'  # 避免保存为npz文件导致record和model的记录不匹配
    log.add_final_state(config=config)  # 写日志

    if (not config.create_new) and config.save_record:  # 如果不是创建的新模型，并且要保存训练记录
        # load previous state, update
        # 以下都是从npz文件中加载同一个模型之前的训练记录，总epochs(int)，历史acc(加载后是数组)
        # Todo: 添加记录loss
        prev_state = np.load(
            os.path.join(config.path, config.save_folder,
                         config.save_name) + '.npz')  # 每次只从上一次有保存过模型的record读取，避免从.tmp.npz读取
        config.epochs = prev_state['epochs'] + config.epochs  # 不能用+=，否则顺序反了
        config.train_record['acc'] = list(prev_state['acc_train']) + config.train_record['acc']
        config.valid_record['acc'] = list(prev_state['acc_valid']) + config.valid_record['acc']
    if config.save_record:  # 如果要保存训练记录，如果保存模型，就是npz文件，如果没有保存模型就是.tmp.npz
        np.savez(
            os.path.join(config.path, config.save_folder, config.record_name),  # 如果没有保存模型，就从.tmp.npz读取，否则从.npz读取
            epochs=config.epochs,
            acc_train=config.train_record['acc'],
            acc_valid=config.valid_record['acc'],
        )
        print(f"Saved Training Process Record to\t{config.record_name}")
        log.add_training_state(config)
        # if log.log_dict['']
    if draw:  # 作图部分
        # Todo: Add draw loss, API should not change
        show_accuracy_through_epoch(config=config, file_name=config.record_name)
        show_pred_true_through_day(
            config=config,
            pred_iter=config.train_dict['pred'] + config.valid_dict['pred'],
            true_iter=config.train_dict['true'] + config.valid_dict['true'],
            div_day=config.div_day,
            sample=config.sample_training,
            demo_ratio=config.demo_ratio,
            ymargin_rate=config.ymargin_rate,
            fig_dpi=config.fig_dpi,
        )


def get_between_days(begin, end, drop_begin=True, drop_end=False):
    """
    获取两天之间的所有日期
    :param begin:时间起点字符串, '2021-10-24'这种
    :param end: 时间终点
    :param drop_begin: 返回的列表是否删去起点
    :param drop_end: 返回的列表是否删去终点
    :return: 返回日期列表
    """
    date_list = []
    begin = datetime.strptime(begin, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    while begin <= end:
        date_str = begin.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin += timedelta(days=1)
    if drop_begin:
        date_list = date_list[1:]
    if drop_end:
        date_list = date_list[:-1]
    return date_list


def run_regression_evaluation(config, target_date, pred_array, true_array):
    # 迭代预测的全过程，直接运行，目前只支持模型为Prices后缀的执行
    config.model.eval()
    data = config.data
    if len(config.select) == 5:
        if config.model_type == 'BP':
            last_line = list(data.true_in[:config.div_day][-1])
            last_day = last_line[-3:]
            date_list = get_between_days(
                begin='-'.join([str(int(x)) for x in last_day]),
                end=target_date,
                drop_begin=True,
                drop_end=False,
            )
            cutoff = len(pred_array) - config.div_day
            pred_array = list(np.array(pred_array)[:-cutoff, :])
            for i, date in enumerate(date_list):
                target_date = [float(x) for x in date.split('-')]
                norm_line = (np.array(last_line) - data.mean_std['train'][0][0]) / data.mean_std['train'][1][0]
                # Todo:modify the prediction, mean, std
                values = config.model(torch.tensor(norm_line).float()).detach()  # predict
                values = np.array(values, dtype=np.float64)
                pred_array.append(values)
                new_features = list(values)
                days = config.days
                num_day_feature = data.proc_dataframe.shape[1]
                # 这里没有使用全部的volume
                prev_vols = [last_line[x] for x in range(num_day_feature - 1, len(last_line), num_day_feature)]
                volume = np.random.normal(loc=np.mean(prev_vols), scale=np.std(prev_vols))  # get volume
                new_features.append(volume)
                new_features.extend(target_date)  # 日期
                last_line = last_line[num_day_feature:] + new_features

            show_pred_true_through_day(
                config=config,
                pred_iter=pred_array[:len(true_array)],
                true_iter=true_array,
                div_day=config.div_day,
                eval=True,
                sample=config.sample_evaluation,
                demo_ratio=config.demo_ratio,
                ymargin_rate=config.ymargin_rate,
                fig_dpi=config.fig_dpi,
            )
        if config.rnn:
            last_floor = data.true_in[:config.div_day][-1]
            last_day = last_floor[-1][:3]
            date_list = get_between_days(
                begin='-'.join([str(int(x)) for x in last_day]),
                end=target_date,
                drop_begin=False,
                drop_end=False,
            )
            cutoff = len(pred_array) - config.div_day
            pred_array = list(np.array(pred_array)[:-cutoff, :])
            for i, date in enumerate(date_list):
                target_date = [float(x) for x in date.split('-')]
                last_floor = np.array(last_floor)
                norm_floor = (last_floor - data.mean_std['train'][0][0]) / data.mean_std['train'][1][0]
                values = config.model(torch.unsqueeze(torch.tensor(norm_floor).float(), dim=0)).detach()
                values = torch.squeeze(values, dim=0)
                values = np.array(values, dtype=np.float64)

                pred_array.append(values)
                new_features = list(values)
                days = config.days
                # 这里没有提取全部的volume
                # 上下计算prev_vols的办法存疑，14天的数据量我考虑偏小了
                prev_vols = last_floor[:, -1:].reshape(-1)
                # print(prev_vols)
                volume = np.random.normal(loc=prev_vols.mean(), scale=prev_vols.std())
                new_features = target_date + new_features
                new_features.append(volume)
                # print(new_features)
                # print(last_floor)
                last_floor = list(last_floor[1:])
                last_floor.append(new_features)
            show_pred_true_through_day(
                config=config,
                pred_iter=pred_array[:len(true_array)],
                true_iter=true_array,
                div_day=config.div_day,
                eval=True,
                sample=config.sample_evaluation,
                demo_ratio=config.demo_ratio,
                ymargin_rate=config.ymargin_rate,
                fig_dpi=config.fig_dpi,
            )


def run_regression_forecast(config, target_date, log, demo_ratio, draw=True):
    # 类似 training，多的不说了
    for t in range(config.epochs):
        print(f"Epoch {t + 1}\n-----------------------------------------")
        tmp_list = train_regression(config.train_dataloader, config.model, config.loss_fn, config.optimizer,
                                    config.device, config.train_dict)
        print(f"Train Error:\n Accuracy: {(100 * tmp_list[0]):>12.2f}%, Avg loss: {tmp_list[1]:>12.5f} \n")
        config.train_record['acc'].append(tmp_list[0]), config.train_record['loss'].append(tmp_list[1])
    print("Done!")
    if config.save_model:
        torch.save(config.model.state_dict(), os.path.join(config.path, config.save_folder, config.save_name))
        print(f"Saved PyTorch Model State to\t\t{config.save_name}")
        config.record_name = config.save_name + '.npz'
    else:
        print("\nDid not save training data")
        config.record_name = config.save_name + '.tmp.npz'  # 避免保存为npz文件导致record和model的记录不匹配
    log.add_final_state(config=config)
    if (not config.create_new) and config.save_record:
        # load previous state, update
        prev_state = np.load(
            os.path.join(config.path, config.save_folder,
                         config.save_name) + '.npz')  # 每次只从上一次有保存过模型的record读取，避免从.tmp.npz读取
        config.epochs = prev_state['epochs'] + config.epochs  # 不能用+=，否则顺序反了
        config.train_record['acc'] = list(prev_state['acc_train']) + config.train_record['acc']
        config.valid_record['acc'] = list(prev_state['acc_valid']) + config.valid_record['acc']
    if config.save_record:
        np.savez(
            os.path.join(config.path, config.save_folder, config.record_name),  # 如果没有保存模型，就从.tmp.npz读取，否则从.npz读取
            epochs=config.epochs,
            acc_train=config.train_record['acc'],
            acc_valid=config.valid_record['acc'],
        )
        print(f"Saved Training Process Record to\t{config.record_name}")
        log.add_trained_state(config)
    pred_array = config.train_dict['pred']
    true_array = config.train_dict['true']

    config.model.eval()
    data = config.data
    if not config.rnn:
        last_line = list(data.true_in[:][-1])
        last_day = last_line[-3:]
        date_list = get_between_days(
            begin='-'.join([str(int(x)) for x in last_day]),
            end=target_date,
            drop_begin=True,
            drop_end=False,
        )
        cutoff = 0
        pred_array = list(np.array(pred_array)[:, :])
        # print(len(pred_array))
        for i, date in enumerate(date_list):
            target_date = [float(x) for x in date.split('-')]
            norm_line = (np.array(last_line) - data.mean_std['full'][0][0]) / data.mean_std['full'][1][0]
            # Todo:modify the prediction, mean, std
            values = config.model(torch.tensor(norm_line).float()).detach()  # predict
            values = np.array(values, dtype=np.float64)
            pred_array.append(values)
            new_features = list(values)
            days = config.days
            num_day_feature = data.proc_dataframe.shape[1]
            # 这里没有使用全部的volume
            prev_vols = [last_line[x] for x in range(num_day_feature - 1, len(last_line), num_day_feature)]
            volume = np.random.normal(loc=np.mean(prev_vols), scale=np.std(prev_vols))  # get volume
            new_features.append(volume)
            new_features.extend(target_date)  # 日期
            last_line = last_line[num_day_feature:] + new_features
        show_forecast(
            config=config,
            pred_iter=pred_array,
            true_iter=true_array,
            date_list=date_list,
            demo_ratio=config.demo_ratio,
            ymargin_rate=config.ymargin_rate,
            xmargin_day=config.xmargin_day,
            fig_dpi=config.fig_dpi,
        )
    else:
        last_floor = data.true_in[:][-1]
        last_day = last_floor[-1][:3]
        date_list = get_between_days(
            begin='-'.join([str(int(x)) for x in last_day]),
            end=target_date,
            drop_begin=False,
            drop_end=False,
        )
        cutoff = 0
        pred_array = list(np.array(pred_array)[:, :])
        for i, date in enumerate(date_list):
            target_date = [float(x) for x in date.split('-')]
            last_floor = np.array(last_floor)
            norm_floor = (last_floor - data.mean_std['full'][0][0]) / data.mean_std['full'][1][0]
            values = config.model(torch.unsqueeze(torch.tensor(norm_floor).float(), dim=0)).detach()
            values = torch.squeeze(values, dim=0)
            values = np.array(values, dtype=np.float64)

            pred_array.append(values)
            new_features = list(values)
            days = config.days
            # 这里没有提取全部的volume
            # 上下计算prev_vols的办法存疑，14天的数据量我考虑偏小了
            prev_vols = last_floor[:, -1:].reshape(-1)
            # print(prev_vols)
            volume = np.random.normal(loc=prev_vols.mean(), scale=prev_vols.std())
            new_features = target_date + new_features
            new_features.append(volume)
            # print(new_features)
            # print(last_floor)
            last_floor = list(last_floor[1:])
            last_floor.append(new_features)
        show_forecast(
            config=config,
            pred_iter=pred_array,
            true_iter=true_array,
            date_list=date_list,
            demo_ratio=config.demo_ratio,
            ymargin_rate=config.ymargin_rate,
            xmargin_day=config.xmargin_day,
            fig_dpi=config.fig_dpi,
        )


if __name__ == '__main__':
    config = Config()
    log = Logger()
    # if not os.path.exists(config.path):
    #     os.makedirs(config.path)
    if not os.path.exists(config.path):  # 创建保存模型的上级路径，这一文件夹下是log.csv和各个模型的文件夹
        os.makedirs(config.path)
        log.create_new_log(config.path, filename='log.csv', detailed=True)
    preparation(config)
    if not os.path.exists(os.path.join(config.path, config.save_folder)):  # 创建保存模型的路径，这个文件夹就是模型名去掉.pth
        os.makedirs(os.path.join(config.path, config.save_folder))

    config.sample_training = min(len(config.data.proc_dataframe), config.sample_training)
    config.sample_evaluation = min(len(config.data.proc_dataframe), config.sample_evaluation)
    if config.task == 'Regression':
        run_regression_training(
            config=config,
            log=log,
            draw=True,
        )
        run_regression_evaluation(
            config=config,
            target_date=config.target_date,
            pred_array=config.train_dict['pred'] + config.valid_dict['pred'],
            true_array=config.train_dict['true'] + config.valid_dict['true'],
        )
    # 以下是加载训练好的模型的部分，不需要上面的运行代码，两部分是独立的，我只是为了方便写在一个py文件里面，原来的pretrained.py可以删了
    config.save_name = ''
    # 'NVAX_BP_20211029_105023_Prices.pth'
    config.save_folder = config.save_name.split('.')[0]
    config.save_model = False
    config.save_record = True
    config.epochs = 20
    config.learning_rate = 1e-3
    config.batch_size = 512
    config.path = os.path.join('ProjectStorage', 'Model', config.task, config.stock_name, config.model_type)
    config.validation = False
    config.div_rate = 1
    config.demo_ratio = 0.7
    config.xmargin_day = 5
    log = Logger()
    config.model_type = config.save_name.split('_')[1]
    config.select = ['Low', 'High', 'Open', 'Close', 'Adj Close'] \
        if config.save_folder.split('_')[-1] == 'Prices' else [config.save_folder.split('_')[-1]]

    # print(config.model_type)
    if config.model_type == 'LSTM' or config.model_type == 'GRU':
        config.rnn = True
    else:
        config.rnn = False

    if not os.path.exists(config.path):
        os.makedirs(config.path)
        log.create_new_log(config.path, filename='log.csv', detailed=True)
    if not os.path.exists(config.path):  # 创建保存模型的上级路径，这一文件夹下是log.csv和各个模型的文件夹
        os.makedirs(config.path)
        log.create_new_log(config.path, filename='log.csv', detailed=True)
    preparation(config)
    if not os.path.exists(os.path.join(config.path, config.save_folder)):  # 创建保存模型的路径，这个文件夹就是模型名去掉.pth
        os.makedirs(os.path.join(config.path, config.save_folder))

    config.sample_training = min(len(config.data.proc_dataframe), config.sample_training)
    config.sample_evaluation = min(len(config.data.proc_dataframe), config.sample_evaluation)
    if config.task == 'Regression':
        run_regression_forecast(
            config=config,
            target_date=config.target_date,
            log=log,
            demo_ratio=config.demo_ratio,
            draw=True,
        )
    # Todo:添加删除当前目录下tmp.npz文件的操作，如果文件存在

    # Todo: 只预测7
    # Todo: 14-28天的数据
    # Todo: 修改Adam作为优化器
    # 修改模型的预测作图
