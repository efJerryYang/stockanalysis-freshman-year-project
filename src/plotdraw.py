import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


def show_accuracy_through_epoch(config, file_name: str, train_color='orange', valid_color='blue', fig_dpi=800):
    """
    :param file_name:
    :param prefix: save_name
    :param postfix: stock feature list
    :param train_color:
    :param valid_color:
    :param fig_dpi:
    :return:d
    """
    file = np.load(os.path.join(config.path, config.save_folder, file_name))
    epochs = file['epochs']  # int
    acc_train, acc_valid = file['acc_train'], file['acc_valid']  # numpy.ndarray, numpy.ndarray
    plt.plot(range(1, epochs + 1), acc_train, train_color, label='training')
    plt.plot(range(1, len(acc_valid) + 1), acc_valid, valid_color, label='validation')

    x_scale, y_scale = [1, epochs + 1], [-1.0, 1.0]
    plt.axis(x_scale + y_scale)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='best')

    time_format = '%Y%m%d_%H%M%S'  # 20211008_140844
    time_postfix = datetime.now().strftime(time_format)
    fig_name_postfix = '_'.join([config.save_name, time_postfix, 'acc_epoch.png'])
    fig_path_1 = os.path.join('ProjectStorage', 'Plot', config.save_folder, 'Accuracy_Loss')
    fig_path_2 = os.path.join(config.path, config.save_folder, 'Accuracy_Loss')
    if not os.path.exists(fig_path_1):
        os.makedirs(fig_path_1)
    if not os.path.exists(fig_path_2):
        os.makedirs(fig_path_2)
    plt.savefig(os.path.join(fig_path_1, fig_name_postfix), dpi=fig_dpi)
    plt.savefig(os.path.join(fig_path_2, fig_name_postfix), dpi=fig_dpi)
    plt.show()


def show_pred_true_through_day(config, pred_iter, true_iter, div_day: int, eval=False,
                               pred_color='deepskyblue', true_color='salmon',
                               sample=100, demo_ratio=0.7, ymargin_rate=0.2,
                               pred_xtrans=0, fig_dpi=800):
    before = int(sample * demo_ratio)
    after = sample - before
    true_iter = true_iter[div_day - before:div_day] + true_iter[div_day:div_day + after]
    pred_iter = pred_iter[div_day - before:div_day] + pred_iter[div_day:div_day + after]
    # plt.plot(range(1, sample + 1), true_iter, color=true_color, label='_'.join(['true'] + config.select))
    # plt.plot(range(1 + pred_xtrans, sample + 1 + pred_xtrans), pred_iter, color=pred_color,
    #          label='_'.join(['pred'] + config.select))
    plt.plot(range(1, sample + 1), np.mean(true_iter, axis=1), color=true_color, label='_'.join(['true', 'avg']))
    plt.plot(range(1 + pred_xtrans, sample + 1 + pred_xtrans), np.mean(pred_iter, axis=1), color=pred_color,
             label='_'.join(['pred', 'avg']))
    t = true_iter + pred_iter
    flat_list = [item for sublist in t for item in sublist]
    # src: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    y_scale = [min(flat_list) * (1 - ymargin_rate),
               max(flat_list) * (1 + ymargin_rate)]
    plt.axvline(before, color='black', ls='--')
    plt.axis([0, sample] + y_scale)
    plt.xlabel('Day')
    plt.legend(loc='best')

    # time postfix
    time_format = '%Y%m%d_%H%M%S'  # 20211008_140844
    time_postfix = datetime.now().strftime(time_format)
    if eval:
        fig_name_postfix = '_'.join([config.save_name, time_postfix, 'eval_true.png'])
    else:
        fig_name_postfix = '_'.join([config.save_name, time_postfix, 'pred_true.png'])
    fig_path_1 = os.path.join('ProjectStorage', 'Plot', config.save_folder, 'Pred_Eval_True')
    fig_path_2 = os.path.join(config.path, config.save_folder, 'Pred_Eval_True')
    if not os.path.exists(fig_path_1):
        os.makedirs(fig_path_1)
    if not os.path.exists(fig_path_2):
        os.makedirs(fig_path_2)
    plt.savefig(os.path.join(fig_path_1, fig_name_postfix), dpi=fig_dpi)
    plt.savefig(os.path.join(fig_path_2, fig_name_postfix), dpi=fig_dpi)
    plt.show()


def show_forecast(config, pred_iter, true_iter, date_list,
                  pred_color='deepskyblue', true_color='salmon', forecast_color='green',
                  demo_ratio=0.7, ymargin_rate=0.2, xmargin_day=5, pred_xtrans=0, fig_dpi=800):
    demo_len = int(np.ceil(len(date_list) / (1 - demo_ratio)))
    true_len = demo_len - len(date_list)

    pred_iter, true_iter = np.array(pred_iter), np.array(true_iter)

    plt.plot(range(1, true_len + 1), np.mean(true_iter[-true_len:, :], axis=1), color=true_color,
             label='_'.join(['true', 'avg']))
    # print(pred_iter[-demo_len:-len(date_list), :])
    # print(pred_iter[-demo_len:-len(date_list), :])
    plt.plot(range(1, true_len + 1), np.mean(pred_iter[-demo_len:-len(date_list), :], axis=1), color=pred_color,
             label='_'.join(['pred', 'avg']))

    plt.plot(range(true_len + 1, demo_len + 1), np.mean(pred_iter[-len(date_list):, :], axis=1), color=forecast_color,
             label='_'.join(['forecast', 'avg']))
    t = list(true_iter[-true_len:, :]) + list(pred_iter[-demo_len:-len(date_list), :])
    flat_list = [item for sublist in t for item in sublist]
    # src: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    y_scale = [min(flat_list) * (1 - ymargin_rate),
               max(flat_list) * (1 + ymargin_rate)]
    plt.axvline(true_len + 0.5, color='black', ls='--')
    plt.axis([0, demo_len + xmargin_day] + y_scale)
    plt.xlabel('Day')
    plt.legend(loc='best')

    time_format = '%Y%m%d_%H%M%S'  # 20211008_140844
    time_postfix = datetime.now().strftime(time_format)
    fig_name_postfix = '_'.join([config.save_name, time_postfix, 'forecast.png'])
    fig_path_1 = os.path.join('ProjectStorage', 'Plot', config.save_folder, 'Evaluation')
    fig_path_2 = os.path.join(config.path, config.save_folder, 'Evaluation')
    if not os.path.exists(fig_path_1):
        os.makedirs(fig_path_1)
    if not os.path.exists(fig_path_2):
        os.makedirs(fig_path_2)
    plt.savefig(os.path.join(fig_path_1, fig_name_postfix), dpi=fig_dpi)
    plt.savefig(os.path.join(fig_path_2, fig_name_postfix), dpi=fig_dpi)
    plt.show()


def show_norm_true(config, sample=50):
    time_format = '%Y%m%d_%H%M%S'  # 20211008_140844
    time_postfix = datetime.now().strftime(time_format)
    dim = str(2) if not config.rnn else str(3)
    axis = '[' + str(config.axis) + ']'
    file_name_prefix = '_'.join([config.stock_name, str(config.div_rate), dim + axis, str(sample)])
    file_name = '_'.join([file_name_prefix, time_postfix])
    path = os.path.join('ProjectStorage', 'StockData', 'Comparison', config.stock_name)
    path_csv = os.path.join(path, 'sample_csv')
    path_fig = os.path.join(path, 'sample_fig' + axis)
    if not os.path.exists(path_csv):
        os.makedirs(path_csv)
    if not os.path.exists(path_fig):
        os.makedirs(path_fig)

    before = int(config.div_rate * sample)
    avg_prices = dict(true=[], norm=[])
    volume = dict(true=[], norm=[])
    avg_prices['true'] = config.data.true_in[config.div_day - before:config.div_day + sample - before, 3:8].mean(
        axis=1)
    volume['norm'] = config.data.norm_in['train_valid_separate'][
                     config.div_day - before:config.div_day + sample - before, 8:9].reshape(-1)
    volume['true'] = config.data.true_in[config.div_day - before:config.div_day + sample - before, 8:9].reshape(-1)
    avg_prices['norm'] = config.data.norm_in['train_valid_separate'][
                         config.div_day - before:config.div_day + sample - before, 3:8].mean(axis=1)
    proc_to_file = np.concatenate([avg_prices['true'].reshape(-1, 1), volume['true'].reshape(-1, 1),
                                   avg_prices['norm'].reshape(-1, 1), volume['norm'].reshape(-1, 1)], axis=1)
    # print(proc_to_file)
    with open(os.path.join(path_csv, file_name + '.csv'), 'w') as f:
        f.write(','.join(['avg_prices_true', 'volume_true', 'avg_prices_norm', 'volume_norm']) + '\n')
        for line in proc_to_file:
            f.write(','.join([str(x) for x in line]) + '\n')
    plt.plot(range(1, sample + 1), volume['norm'], color='deepskyblue', label='volume_norm')
    flat_list = volume['norm']
    ymargin = (max(flat_list) - min(flat_list)) * config.ymargin_rate / 2
    y_scale = [min(flat_list) - ymargin,
               max(flat_list) + ymargin]
    print(y_scale)
    plt.axvline(before + 0.5, color='black', ls='--')
    plt.axis([0, sample + 1] + y_scale)
    plt.xlabel('Day')
    plt.legend(loc='best')
    fig_name_postfix = '_'.join([file_name, 'volume_norm.png'])
    plt.savefig(os.path.join(path_fig, fig_name_postfix), dpi=config.fig_dpi)
    plt.show()

    plt.plot(range(1, sample + 1), volume['true'], color='blue', label='volume_true')
    flat_list = volume['true']
    ymargin = (max(flat_list) - min(flat_list)) * config.ymargin_rate / 2
    y_scale = [min(flat_list) - ymargin,
               max(flat_list) + ymargin]
    print(y_scale)
    plt.axvline(before + 0.5, color='black', ls='--')
    plt.axis([0, sample + 1] + y_scale)
    plt.xlabel('Day')
    plt.legend(loc='best')
    fig_name_postfix = '_'.join([file_name, 'volume_true.png'])
    plt.savefig(os.path.join(path_fig, fig_name_postfix), dpi=config.fig_dpi)
    plt.show()

    plt.plot(range(1, sample + 1), avg_prices['norm'], color='salmon', label='avg_prices_norm')
    flat_list = avg_prices['norm']
    ymargin = (max(flat_list) - min(flat_list)) * config.ymargin_rate / 2
    y_scale = [min(flat_list) - ymargin,
               max(flat_list) + ymargin]
    print(y_scale)
    plt.axvline(before + 0.5, color='black', ls='--')
    plt.axis([0, sample + 1] + y_scale)
    plt.xlabel('Day')
    plt.legend(loc='best')
    fig_name_postfix = '_'.join([file_name, 'avg_prices_norm.png'])
    plt.savefig(os.path.join(path_fig, fig_name_postfix), dpi=config.fig_dpi)
    plt.show()

    plt.plot(range(1, sample + 1), avg_prices['true'], color='red', label='avg_prices_true')
    flat_list = avg_prices['true']
    ymargin = (max(flat_list) - min(flat_list)) * config.ymargin_rate / 2
    y_scale = [min(flat_list) - ymargin,
               max(flat_list) + ymargin]
    print(y_scale)
    plt.axvline(before + 0.5, color='black', ls='--')
    plt.axis([0, sample + 1] + y_scale)
    plt.xlabel('Day')
    plt.legend(loc='best')
    fig_name_postfix = '_'.join([file_name, 'avg_prices_true.png'])
    plt.savefig(os.path.join(path_fig, fig_name_postfix), dpi=config.fig_dpi)
    plt.show()
