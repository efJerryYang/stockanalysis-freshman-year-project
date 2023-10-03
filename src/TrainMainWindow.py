from ModelList import Ui_ModelList
import wx
import wx.grid
import csv
import os
import pandas as pd
import numpy as np
from DataProcess import DataProcess


def exercise():
    class Ui_MainWindow(wx.Frame):
        def __init__(self):
            self.origin_path = os.getcwd()
            self.path = os.path.join(os.getcwd(), 'ProjectStorage', 'Log')
            # 建窗口
            wx.Frame.__init__(self, None, -1, title='训练数据查看', size=(950, 820))
            self.Center()  # 让窗口居中
            self.panel = wx.Panel(self)
            DataProcess(0)
            li_col0 = ['待测试股票列表', '模型类别概览', '任务', '上一次测试时间', '最高正确率', '平均正确率']
            self.grid0 = wx.grid.Grid(self.panel)
            self.grid0.CreateGrid(50, 6)  # 建立表格num x 6
            self.grid0.EnableEditing(False)  # 只读

            with open(os.path.join(self.path, 'MaxMeanTrainAcc.csv')) as f:
                f_csv = csv.reader(f)

                headers = next(f_csv)
                SData = []  # 创建一个二维数组记录数据
                for row in f_csv:
                    SData.append(row)
                num = len(SData)  # num为Sdata行数
                # print(num)
                # print(SData)    #从SData[2][0]开始
                for col in range(6):
                    self.grid0.SetColLabelValue(col, li_col0[col])  # 列坐标
                for row in range(num):  # 讲单元格数据填入表格
                    for col in range(6):
                        if row >= 2:
                            self.grid0.SetCellValue(row - 2, col, SData[row][col])

            self.grid0.AutoSize()  # 设置行和列自动调整
            for col in range(2, 6):
                self.grid0.SetColSize(col, 160)

            DataProcess(1)
            li_col1 = ['可选用股票列表', '模型类别概览', '任务', '上一次测试时间', '最高正确率', '平均正确率']
            self.grid1 = wx.grid.Grid(self.panel)
            self.grid1.CreateGrid(50, 6)  # 建立表格num x 6
            self.grid1.EnableEditing(False)  # 只读
            with open(os.path.join(self.path, 'MaxMeanTrainAcc.csv')) as f:
                f_csv = csv.reader(f)

                headers = next(f_csv)
                SData = []  # 创建一个二维数组记录数据
                for row in f_csv:
                    SData.append(row)
                num = len(SData)  # num为Sdata行数
                # print(num)
                # print(SData)    #从SData[2][0]开始
                for col in range(6):
                    self.grid1.SetColLabelValue(col, li_col1[col])  # 列坐标
                for row in range(num):  # 将单元格数据填入表格
                    for col in range(6):
                        if row >= 2:
                            self.grid1.SetCellValue(row - 2, col, SData[row][col])

            self.grid1.AutoSize()  # 设置行和列自动调整
            for col in range(2, 6):
                self.grid1.SetColSize(col, 160)

            # 使用gridsizer布局，使得frame在上下分别显示grid1和grid2
            gridsizer = wx.GridSizer(cols=1, rows=2, vgap=10, hgap=100)  # 布局2 x 1
            # title0 = wx.StaticText(self.panel, label='Title0xx',pos=(15,8))   #手动调节位置挪到中央，，，
            # title1 = wx.StaticText(self.panel, label='Title1xxx',pos=(15,140))
            gridsizer.Add(self.grid1, 1, wx.EXPAND)
            gridsizer.Add(self.grid0, 1, wx.EXPAND)
            self.panel.SetSizer(gridsizer)

            # 绑定事件，产生子窗口
            ##wx.grid.EVT_GRID_CELL_LEFT_DCLICK  在单元格双击左键
            self.Bind(wx.grid.EVT_GRID_CELL_LEFT_DCLICK, self.UpOpenDetail)
            # self.Bind(wx.grid.EVT_GRID_LABEL_LEFT_DCLICK,self.DownOpenDetail)

        # 获得行序列和数据打开相关子窗口，注意：在上方表格左键双击单元格，在下方表格左键双击行标签（目前我只能这样区分上下两个表格）
        def UpOpenDetail(self, event):
            # f0 = open('StockName_Select.txt', 'r')  # 读取txt文件中的股票名称并且存入列表li-name0中
            # li_name0 = f0.readlines()
            row_num = event.GetRow()  # 获取点击的行序号，从零开始
            # print(row_num)

            if self.grid0.IsSelection():  ##选择表0
                DataProcess(0)
                f = open(os.path.join(self.path, 'MaxMeanTrainAcc.csv'), 'r')
                f_csv = csv.reader(f)
                headers = next(f_csv)
                SData = []  # 创建一个二维列表记录数据  #从SData[2][0]开始
                for row in f_csv:
                    SData.append(row)

                DataProcess(2)
                df = pd.read_csv(os.path.join(self.path, 'MaxMeanTrainAcc.csv'), encoding='utf-8')
                df1 = df[df['Stock Name'] == SData[row_num + 2][0]]  # 筛选出需要的股票

                grouped = df1.groupby(by=['Stock Name', 'Model Name', 'Model Type', 'Task'])
                MaxMeanData = grouped.agg({'Time': np.max, 'Valid Acc': [np.max, np.mean], 'Div Rate': np.max})

                MaxMeanData.to_csv(os.path.join(self.path, 'MaxMeanTrainAcc.csv'), encoding='utf-8',
                                   index=True)  # 填入子窗口的数据，存入此路径下的MaxMeanTrainAcc.csv
                f.close()
                dialog = Ui_ModelList()  # 创建子窗口实例
                dialog.Show()  # 显示子窗口
                event.Skip()
            if self.grid1.IsSelection():  ##选择表1

                DataProcess(1)
                f = open(os.path.join(self.path, 'MaxMeanTrainAcc.csv'), 'r')
                f_csv = csv.reader(f)
                headers = next(f_csv)
                SData = []  # 创建一个二维列表记录数据  #从SData[2][0]开始
                for row in f_csv:
                    SData.append(row)

                DataProcess(2)
                df = pd.read_csv(os.path.join(self.path, 'MaxMeanTrainAcc.csv'), encoding='utf-8')
                df1 = df[df['Stock Name'] == SData[row_num + 2][0]]  # 筛选出需要的股票

                grouped = df1.groupby(by=['Stock Name', 'Model Name', 'Model Type', 'Task'])
                MaxMeanData = grouped.agg({'Time': np.max, 'Valid Acc': [np.max, np.mean], 'Div Rate': np.max})

                MaxMeanData.to_csv(os.path.join(self.path, 'MaxMeanTrainAcc.csv'), encoding='utf-8',index=True)
                # 填入子窗口的数据，存入此路径下的MaxMeanTrainAcc.csv
                f.close()

                dialog = Ui_ModelList()  # 创建子窗口实例
                dialog.Show()  # 显示子窗口
                event.Skip()

    app = wx.App()
    frame = Ui_MainWindow()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    exercise()
