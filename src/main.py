import wx
import wx.grid
import Prediction_mode
import TrainMainWindow
import spider
import re
import os
import time
# import pandas as pd
import numpy as np
import yfinance as yf
import datetime


class Frame(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent=None, id=-1, title="股票预测—Beta", size=(565, 500))
        self.Center()  # 设置窗口居中
        panel = wx.Panel(self)  # 创建面板
        panel.SetBackgroundColour('White')

        col_list = ['股票名称', '简称', '最后更新时间']
        self.grid = wx.grid.Grid(panel)

        file_names = []
        path = os.path.join('ProjectStorage', 'StockData', 'Full')
        for file in os.listdir(path):
            if file.split('.')[1] == 'csv':
                file_names.append(file)
        save_dict = np.load(os.path.join(path, 'stock_dict.npy'), allow_pickle=True).item()

        self.grid.CreateGrid(len(file_names), 3)
        self.grid.EnableEditing(False)
        # self.grid.AutoSize()  # 设置行和列自动调整
        self.grid.SetColSize(0, 260)
        self.grid.SetColSize(1, 60)
        self.grid.SetColSize(2, 130)
        path = os.path.join(os.getcwd(), 'ProjectStorage', 'StockData', 'Full')
        for col in range(3):
            self.grid.SetColLabelValue(col, col_list[col])
        for row, file_csv in enumerate(file_names):
            MTime = time.localtime(os.stat(os.path.join(path, file_csv)).st_mtime)
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', MTime)
            # time_list.append(mtime)
            abbreviation = file_csv.split('.')[0]
            self.grid.SetCellValue(row, 0, save_dict[abbreviation])
            self.grid.SetCellValue(row, 1, abbreviation)
            self.grid.SetCellValue(row, 2, mtime)

        # 模式选择还是弹出一个选择的窗口，因为往外挪有点麻烦
        self.bt_choose = wx.Button(panel, label="模式选择")
        # 为“模式选择”按钮添加事件处理
        self.bt_choose.Bind(wx.EVT_BUTTON, lambda e: ModelChoose(self, -1))

        # 更新股票数据的按钮在这里，self.on_bt_update是这个按钮所绑定的事件函数名称，见131行
        self.bt_update = wx.Button(panel, label="更新股票数据")
        # 为“更新已下载模块”按钮添加事件处理
        self.bt_update.Bind(wx.EVT_BUTTON, self.on_bt_update)

        self.st_relate = wx.StaticText(panel, label="您选择的股票没有合适的模型？\n我们将尽快添加您需要的预测模型。")

        self.bt_relate = wx.Button(panel, label="联系我们")
        # 为“联系我们”按钮添加事件处理
        self.bt_relate.Bind(wx.EVT_BUTTON, lambda e: RelateFrame(self, -1))

        self.st_relate_ = wx.StaticText(panel, label="或者您可以选择下载股票后训练模型")

        self.st_stock_name = wx.StaticText(panel, label="搜索股票名称:")
        self.tc_stock_name = wx.TextCtrl(panel, style=wx.TE_LEFT)

        # 用户输入股票名称后点击“搜索”，self.on_bt_query是这个按钮所绑定的事件的函数名称，见111行
        self.bt_query = wx.Button(panel, label="搜索")
        # 为“查询”按钮添加事件处理
        self.bt_query.Bind(wx.EVT_BUTTON, self.on_bt_query)
        # 添加容器，容器内控件横向排列
        h_sizer_button = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer_button.Add(self.st_stock_name, 0, flag=wx.ALL, border=10)
        h_sizer_button.Add(self.tc_stock_name, 1, flag=wx.ALL, border=10)
        h_sizer_button.Add(self.bt_query, 0, flag=wx.ALL, border=10)

        h_sizer_update = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer_update.Add(self.bt_update, 0, flag=wx.ALL, border=10)
        h_sizer_update.Add(self.bt_choose, 0, flag=wx.ALL, border=10)

        h_sizer_choose = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer_choose.Add(self.st_relate, 0, flag=wx.ALL, border=10)
        h_sizer_choose.Add(self.bt_relate, 0, flag=wx.ALL, border=10)
        h_sizer_choose.Add(self.st_relate_, 0, flag=wx.ALL, border=15)

        # 添加容器，容器内控件纵向排列
        v_sizer_all = wx.BoxSizer(wx.VERTICAL)
        v_sizer_all.Add(h_sizer_update, 1, flag=wx.EXPAND, border=20)
        v_sizer_all.Add(h_sizer_choose, 1, flag=wx.EXPAND, border=20)
        v_sizer_all.Add(h_sizer_button, 1, flag=wx.EXPAND, border=20)

        gridsizer = wx.GridSizer(cols=1, rows=2, vgap=10, hgap=100)
        gridsizer.Add(self.grid, 1, wx.EXPAND)
        gridsizer.Add(v_sizer_all, 1, wx.EXPAND)

        panel.SetSizer(gridsizer)

    # 添加有关事件

    def on_bt_query(self, event):
        path = os.path.join('ProjectStorage', 'StockData', 'Full')
        if not os.path.exists(os.path.join(path, 'stock_dict.npy')):
            save_npy = {}
        else:
            save_npy = np.load(os.path.join(path, 'stock_dict.npy'), allow_pickle=True).item()
        self.name = self.tc_stock_name.GetValue()
        if self.name == "":
            message = '股票名称不能为空'
            wx.MessageBox(message)
        else:
            print(f'Searching\t{self.name}', end=' ')
            spider.StockName_DataSource_DownloadPath[0] = self.name
            abbreviation = spider.Yahoofinance_spider_download(spider.StockName_DataSource_DownloadPath)
            company = spider.StockName_DataSource_DownloadPath[0]
            save_npy.update({abbreviation: company})
            np.save(os.path.join(path, 'stock_dict.npy'), save_npy, allow_pickle=True)
            print(f'\nDownload\t{abbreviation}.csv\tsuccessfully.\nCompany name:\t{company}')

            # spider.StockName_DataSource_DownloadPath.append(path)
            # spider.Yahoofinance_spider_download(spider.StockName_DataSource_DownloadPath)
            # stock_name = spider.StockName_DataSource_DownloadPath[0]
            stock_name = f'{company} ({abbreviation})'
            message = ' '.join(("您所选择的股票为：\n股票名称：", stock_name, "\n数据源：", 'https://finance.yahoo.com/'))
            wx.MessageBox(message)
            p = re.compile(r'[(](.*)[)]', re.S)
            re.findall(p, stock_name)
            self.name = re.findall(p, stock_name)[0]

    def on_bt_update(self, event):
        file_names = []
        path = os.path.join('ProjectStorage', 'StockData', 'Full')
        for file in os.listdir(path):
            if file.split('.')[1] == 'csv':
                file_names.append(file)
        if not os.path.exists(os.path.join(path, 'stock_dict.npy')):
            save_npy = {}
        else:
            save_npy = np.load(os.path.join(path, 'stock_dict.npy'), allow_pickle=True).item()
        for file in file_names:
            modified_time = time.localtime(os.stat(os.path.join(path, file)).st_mtime)
            mTime = time.strftime('%Y-%m-%d', modified_time)
            if mTime != datetime.datetime.now().strftime('%Y-%m-%d'):
                print(f'Updating\t{file}', end=' ')
                print('.', end='')
                file_name = file.split('.')[0]
                tmp = yf.Ticker(file_name)
                tmp_info = tmp.info
                print('.', end='')
                full_name = tmp_info['shortName']
                tmp_csv = tmp.history(period='max')
                tmp_csv.to_csv(os.path.join(path, file))
                save_npy.update({file_name: full_name})
                np.save(os.path.join(path, 'stock_dict.npy'), save_npy, allow_pickle=True)
                print('.', end='')
                print(f'\nUpdate\t\t{file}\tsuccessfully.')
            #     print(f'Updating\t{file}', end=' ')
            #     spider.StockName_DataSource_DownloadPath[0] = file_name
            #     spider.Yahoofinance_spider_download(spider.StockName_DataSource_DownloadPath)
            #     company = spider.StockName_DataSource_DownloadPath[0]
            #     save_npy.update({file_name: company})
            else:
                print(f'{file}\thas already been up-to-date.')
        wx.MessageBox('所有数据均已更新完成！')


class ModelChoose(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title="模式选择", size=(300, 200))
        self.Center()
        panel = wx.Panel(self)
        panel.SetBackgroundColour('white')
        bt_exercise = wx.Button(panel, label="训练模式—可查看测试集和训练集结果")
        bt_exercise.SetForegroundColour('black')
        bt_exercise.SetBackgroundColour('white')
        bt_exercise.Bind(wx.EVT_BUTTON, lambda e: TrainMainWindow.exercise())

        bt_predict = wx.Button(panel, label="预测模式—可查看目标日期的股票价格")
        bt_predict.SetForegroundColour('black')
        bt_predict.SetBackgroundColour('white')
        bt_predict.Bind(wx.EVT_BUTTON, lambda e: Prediction_mode.predict())
        # 添加容器，容器内控件纵向排列
        v_sizer = wx.BoxSizer(wx.VERTICAL)
        v_sizer.Add(bt_exercise, 0, flag=wx.ALL, border=20)
        v_sizer.Add(bt_predict, 0, flag=wx.ALL, border=20)
        panel.SetSizer(v_sizer)
        self.Show(True)

    def on_bt_predict(self, event):
        pass


class RelateFrame(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title="联系我们", size=(300, 200))
        self.Center()
        panel = wx.Panel(self)
        panel.SetBackgroundColour('white')

        self.st_email_name = wx.StaticText(panel, label="您的联系方式:")
        self.tc_email_name = wx.TextCtrl(panel, style=wx.TE_LEFT)

        bt_send = wx.Button(panel, label="发送")
        bt_send.Bind(wx.EVT_BUTTON, self.bt_email)
        bt_send.SetForegroundColour('black')
        bt_send.SetBackgroundColour('white')

        self.tc_email = wx.TextCtrl(panel, style=wx.TE_MULTILINE, size=(250, 100))
        # 添加容器，容器内控件横向排列
        h_sizer_email = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer_email.Add(self.st_email_name, 0, flag=wx.ALL, border=11)
        h_sizer_email.Add(self.tc_email_name, 1, flag=wx.ALL, border=10)
        h_sizer_email.Add(bt_send, 0, flag=wx.ALL, border=10)

        h_sizer_email_content = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer_email_content.Add(self.tc_email, 0, flag=wx.ALL | wx.EXPAND, border=10)

        # 添加容器，容器内控件纵向排列
        v_sizer = wx.BoxSizer(wx.VERTICAL)
        v_sizer.Add(h_sizer_email, 1, flag=wx.ALL, border=5)
        v_sizer.Add(h_sizer_email_content, 2, flag=wx.ALL, border=5)
        panel.SetSizer(v_sizer)
        self.Show(True)

    def bt_email(self, event):
        user_email = self.tc_email_name.GetValue()
        # print(user_email)
        content = self.tc_email.GetValue()
        # print(content)
        # print("vdfhbfgnfmjff")
        import zmail
        # 你的邮件内容
        mail_content = {
            'subject': 'Project: StockAnalysis2020',  # 邮件标题写在这
            'content_text': f'Reply from {user_email}\n\n{content}\n',  # 邮件正文写在这
        }

        # 使用你的邮件账户名和密码登录服务器
        server = zmail.sever("xxxxxx@xx.xx","xxxxxxxxxxxxx") # 原代码中为我的邮箱和授权码，此处隐去
        # 发送邮件指令
        server.send_mail(["efJerryYang@outlook.com"], mail_content)
        wx.MessageBox('发送成功！')
        self.Close()


class MyApp(wx.App):
    def OnInit(self):
        # 创建窗口对象
        frame = Frame(parent=None, id=-1)
        frame.Show()
        return True

    def OnExit(self):
        print("应用程序退出")
        return 0


if __name__ == '__main__':
    app = MyApp()  # 创建自定以对象App
    app.MainLoop()  # 进入事件主循环
