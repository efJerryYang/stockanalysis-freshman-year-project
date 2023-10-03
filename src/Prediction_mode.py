import wx
import wx.grid
import datetime
import wx.adv
import re
# import pandas
# import numpy as np
# import spider
# import model
import matplotlib
import os
# import sys
import model
# import random


def predict():
    # matplotlib采用WXAgg为后台,将matplotlib嵌入wxPython中
    matplotlib.use("WXAgg")
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
    from matplotlib.backends.backend_wx import NavigationToolbar2Wx
    from matplotlib.figure import Figure
    config = model.Config
    if not os.path.exists(config.path):
        os.makedirs(config.path)

    class PredictFrame(wx.Frame):
        def __init__(self, parent, id):

            self.stock_name = 'APPL'
            self.prediction_model = 'AAPL_BP_20211009_223632.pth'
            wx.Frame.__init__(self, parent=None, id=-1, title="预测模式", size=(950, 820))
            self.Center()  # 设置窗口居中
            self.splitter = wx.SplitterWindow(self)
            self.left_panel = wx.Panel(self.splitter, style=wx.SUNKEN_BORDER)
            self.left_panel.SetBackgroundColour('White')
            self.right_panel = wx.Panel(self.splitter, style=wx.SUNKEN_BORDER)
            self.right_panel.SetBackgroundColour('White')
            self.left_panel.Hide()
            self.right_panel.Hide()
            self.splitter.SplitVertically(self.left_panel, self.right_panel, 0)
            self.splitter.SetMinimumPaneSize(400)
            # 左半面板##################################################################################################
            path1 = os.path.join('ProjectStorage', 'Model', 'Regression')
            list_stock = []
            for file in os.listdir(path1):
                list_stock.append(file)
            self.cb_stock = wx.ComboBox(self.left_panel, -1, value='选择股票', choices=list_stock, style=wx.CB_SORT)
            # 为”选择股票“下拉框添加事件
            self.cb_stock.Bind(wx.EVT_COMBOBOX, self.on_cb_stock)

            self.bt_model = wx.Button(self.left_panel, label='选择模型')
            # 为”选择模型“下拉框添加事件
            self.bt_model.Bind(wx.EVT_BUTTON, lambda e: PModelChoose(self, -1))

            self.st_last_day = wx.StaticText(self.left_panel, label="预测时间的终点：")

            self.date_data = datetime.datetime.now().date()
            self.date_str = self.date_data.strftime('%Y-%m-%d')
            self.date_text = wx.TextCtrl(self.left_panel, -1, self.date_str, style=wx.TE_CENTER,
                                         validator=wx.DefaultValidator)
            self.date_button = wx.Button(self.left_panel, label='选择日期')
            self.date_button.SetForegroundColour('black')
            self.date_button.SetBackgroundColour('white')
            self.date_button.Bind(wx.EVT_BUTTON, lambda e: Calendar(self, -1, self.date_text, 0))

            self.bt_check = wx.Button(self.left_panel, label="查看模型训练数据")
            # 为“查看模型训练数据”按钮添加事件处理
            self.bt_check.Bind(wx.EVT_BUTTON, self.on_bt_check)

            self.bt_predict = wx.Button(self.left_panel, label="预测并作图")
            # 为“预测”按钮添加事件处理
            self.bt_predict.Bind(wx.EVT_BUTTON, self.on_bt_predict_draw)

            self.tc_printing = wx.TextCtrl(self.left_panel, style=wx.TE_MULTILINE)

            # 右半面板################################################################################################

            # #####################################################还需继续添加控件和内容

            # 添加容器，容器内控件横向排列
            h_sizer_choose = wx.BoxSizer(wx.HORIZONTAL)
            h_sizer_choose.Add(self.cb_stock, 1, flag=wx.ALL, border=20)
            h_sizer_choose.Add(self.bt_model, 1, flag=wx.ALL, border=20)

            h_sizer_date = wx.BoxSizer(wx.HORIZONTAL)
            h_sizer_date.Add(self.st_last_day, 1, flag=wx.ALL, border=20)
            h_sizer_date.Add(self.date_text, 1, flag=wx.ALL, border=20)
            h_sizer_date.Add(self.date_button, 1, flag=wx.ALL, border=20)

            h_sizer_predict = wx.BoxSizer(wx.HORIZONTAL)
            h_sizer_predict.Add(self.bt_check, 1, flag=wx.ALL, border=20)
            h_sizer_predict.Add(self.bt_predict, 1, flag=wx.ALL, border=20)

            h_sizer_printing = wx.BoxSizer(wx.HORIZONTAL)
            h_sizer_printing.Add(self.tc_printing, 1, flag=wx.ALL | wx.EXPAND, border=20)

            # 添加容器，容器内控件纵向排列
            v_sizer_all = wx.BoxSizer(wx.VERTICAL)
            v_sizer_all.Add(h_sizer_choose, 1, flag=wx.EXPAND, border=20)
            v_sizer_all.Add(h_sizer_date, 1, flag=wx.EXPAND, border=20)
            v_sizer_all.Add(h_sizer_predict, 1, flag=wx.EXPAND, border=20)
            v_sizer_all.Add(h_sizer_printing, 5, flag=wx.EXPAND, border=20)
            self.left_panel.SetSizer(v_sizer_all)

        def on_cb_stock(self, event):
            self.stock_name = event.GetString()
            config.stock_name = self.stock_name
            print(f'选择的股票简称为: {config.stock_name}')

        def on_bt_date(self, event):
            self.date_text = self.date_text.GetValue()
            print(f'预测的目标日期为: {self.date_text}')

        def on_bt_check(self, event):
            path2 = os.path.join(config.path, 'log.csv')
            with open(path2, 'r') as file:
                message_all = file.readlines()
                text = ''
                for message in message_all:
                    pattern = r'[,]'
                    result = re.split(pattern, message)
                    if result[2] == config.save_name:
                        text = text + message + '\n'
                # print(text)
                self.tc_printing.SetValue(text)

            name = re.split(r'\.', config.save_name)
            # filename_list = []
            picture_name_acc = []
            picture_name_pred = []
            picture_name_eval = []
            path11 = os.path.join('ProjectStorage', 'Model', 'Regression', config.stock_name)
            tuples = os.walk(path11, False)
            for tuple in tuples:
                filenames = tuple[-1]
                for filename in filenames:
                    filename_split = re.split(r'\.', filename)
                    if filename_split[0] == name[0] and filename_split[-1] == 'png':
                        filename_path = list(tuple)
                        picture_name = os.path.join(filename_path[0], filename)
                        picture_name_split = re.split(r'_', picture_name)
                        # print(picture_name_split)
                        if picture_name_split[-2] == 'acc':
                            picture_name_acc.append(picture_name)
                        if picture_name_split[-2] == 'pred':
                            picture_name_pred.append(picture_name)
                        if picture_name_split[-2] == 'eval':
                            picture_name_eval.append(picture_name)
            model_name = [picture_name_acc[-1], picture_name_pred[-1], picture_name_eval[-1]]
            fgs = wx.FlexGridSizer(cols=1, hgap=5, vgap=5)
            for name in model_name:
                img1 = wx.Image(name, wx.BITMAP_TYPE_ANY)
                w = img1.GetWidth()
                h = img1.GetHeight()
                img2 = img1.Scale(w / 10, h / 15)
                sb2 = wx.StaticBitmap(self.right_panel, -1, wx.Bitmap(img2))
                fgs.Add(sb2)
            self.right_panel.SetSizerAndFit(fgs)
            self.Fit()

        def cb_get_model(self, event):
            config.save_name = event.GetString()
            config.save_folder = config.save_name.split('.')[0]
            print(f'选择的模型为: {config.save_name}')

        def on_bt_predict_draw(self, event):
            config.save_model = False
            config.save_record = True
            config.path = os.path.join('ProjectStorage', 'Model', config.task, config.stock_name, config.model_type)
            config.validation = False
            config.div_rate = 1
            config.demo_ratio = 0.7
            config.xmargin_day = 5
            log = model.Logger()
            config.model_type = config.save_name.split('_')[1]
            config.select = ['Low', 'High', 'Open', 'Close', 'Adj Close'] \
                if config.save_folder.split('_')[-1] == 'Prices' else [config.save_folder.split('_')[-1]]

            # print(config.model_type)
            if config.model_type == 'LSTM' or config.model_type == 'GRU':
                config.rnn = True
                config.epochs = 512
                config.learning_rate = 6.4e-4
                config.batch_size = 128
            else:
                config.rnn = False
                config.epochs = 200
                config.learning_rate = 1.6e-4
                config.batch_size = 128

            if not os.path.exists(config.path):
                os.makedirs(config.path)
                log.create_new_log(config.path, filename='log.csv', detailed=True)
            if not os.path.exists(config.path):  # 创建保存模型的上级路径，这一文件夹下是log.csv和各个模型的文件夹
                os.makedirs(config.path)
                log.create_new_log(config.path, filename='log.csv', detailed=True)
            model.preparation(config)
            if not os.path.exists(os.path.join(config.path, config.save_folder)):  # 创建保存模型的路径，这个文件夹就是模型名去掉.pth
                os.makedirs(os.path.join(config.path, config.save_folder))

            config.sample_training = min(len(config.data.proc_dataframe), config.sample_training)
            config.sample_evaluation = min(len(config.data.proc_dataframe), config.sample_evaluation)
            if config.task == 'Regression':
                model.run_regression_forecast(
                    config=config,
                    target_date=config.target_date,
                    log=log,
                    demo_ratio=config.demo_ratio,
                    draw=True,
                )

    class PModelChoose(wx.Frame):
        def __init__(self, parent, id):
            wx.Frame.__init__(self, parent, id, title="模式选择", size=(300, 100))
            self.Center()
            panel = wx.Panel(self)
            panel.SetBackgroundColour('white')
            path4 = os.path.join('ProjectStorage', 'Model', 'Regression', config.stock_name)
            list_model = []
            for files in os.listdir(path4):
                # print(files)
                path5 = os.path.join('ProjectStorage', 'Model', 'Regression', config.stock_name, files)
                for root, dirs, file in os.walk(path5, topdown=True):
                    for file_ in file:
                        if file_.endswith('.pth'):
                            list_model.append(file_)
            cb_model = wx.ComboBox(panel, value="选择模型", choices=list_model, style=wx.CB_SORT)
            cb_model.Bind(wx.EVT_COMBOBOX, self.cb_get_model)

            # 添加容器，容器内控件纵向排列
            v_sizer = wx.BoxSizer(wx.VERTICAL)
            v_sizer.Add(cb_model, 0, flag=wx.ALL, border=20)
            panel.SetSizer(v_sizer)

            self.Show(True)

        def cb_get_model(self, event):
            config.save_name = event.GetString()
            config.save_folder = config.save_name.split('.')[0]
            config.model_type = config.save_name.split('_')[1]
            config.path = os.path.join('ProjectStorage', 'Model', config.task, config.stock_name,
                                       config.model_type)  # 保存的路径，目前修改

            print(f'选择的模型为: {config.save_name}')
            self.Close()

    class Calendar(wx.Frame):
        def __init__(self, parent, id, *p):
            super().__init__(parent, id)
            self.Center()
            self.date_init(p)
            self.Show(True)

        def date_init(self, *p):
            d_time = p[0][0]
            cur_date = datetime.date.today()
            yy = cur_date.year
            if len(p[0]) > 1:
                yy = yy + p[0][1]
            wxc = wx.adv.CalendarCtrl(self, -1, pos=(0, 0), size=(220, 220),
                                      date=datetime.datetime(yy, cur_date.month, cur_date.day),
                                      style=wx.adv.DP_ALLOWNONE)
            wxc.Bind(wx.adv.EVT_CALENDAR, lambda e: self.get_data(wxc, d_time))
            self.SetSize((260, 260))
            self.SetMaxSize((260, 260))
            self.SetTitle('选择日期')

        def get_data(self, handle, t):
            ss = handle.GetDate()
            s = '%s-%s-%s' % (ss.year, ss.month + 1, ss.day)
            t.SetValue(s)
            self.Close()
            config.target_date = s
            print(f'目标日期为: {s}')

    class MyApp(wx.App):
        def OnInit(self):
            frame = PredictFrame(parent=None, id=-1)
            frame.Show()
            return True

        def OnExit(self):
            print("应用程序退出")
            return 0

    app = MyApp()  # 创建自定以对象App
    app.MainLoop()  # 进入事件主循环


if __name__ == '__main__':
    predict()
