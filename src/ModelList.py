'''子窗口一模型列表函数'''

import wx
import wx.grid
import os
import csv
# import numpy as np
# import pandas as pd


# import model
class Ui_ModelList(wx.Frame):
    def __init__(self):

        self.path = os.path.join(os.getcwd(), 'ProjectStorage', 'Log')
        f = open(os.path.join(self.path, 'MaxMeanTrainAcc.csv'), 'r')
        f_csv = csv.reader(f)
        CData = []
        for row in f_csv:
            CData.append(row)
        # li_name = f.readlines()
        num = len(CData)
        wx.Frame.__init__(self, None, title=CData[3][0], size=(950, 820))  # CData[3][0]为所选股票名称
        self.Center()
        self.panel = wx.Panel(self)
        li_col = ['模型列表', '模型类别', '任务', '上一次测试时间', '训练集预测表现', '验证集预测表现', '数据集划分', '是否已添加']
        self.grid = wx.grid.Grid(self.panel)
        self.grid.CreateGrid(num + 50, 8)  # 建立表格num x 7
        self.grid.EnableEditing(False)
        for col in range(8):
            self.grid.SetColLabelValue(col, li_col[col])  # 列坐标
        for row in range(num):  # 讲单元格数据填入表格
            for col in range(7):
                if row >= 3:
                    self.grid.SetCellValue(row - 3, col, CData[row][col + 1])

        '''判断是否trained'''
        df = open(os.path.join(self.path, 'trained.csv'), 'r')
        df_csv = csv.reader(df)
        Check = []
        for row in df_csv:
            Check.append(row)
        c_num = len(Check)
        # row=3
        # while (row>=3)and(row<num):

        for row in range(3, num):
            flag = 0  # flag为0则表示否，为1则表示是
            for c_row in range(1, c_num):
                if (Check[c_row][1] == CData[row][1]):  # 判断模型是否已经添加进入trained
                    # self.grid.SetCellValue(row - 3, 6, '是')
                    flag = 1
                    break
            if (flag == 1):
                self.grid.SetCellValue(row - 3, 7, '是')
            if (flag == 0):
                self.grid.SetCellValue(row - 3, 7, '否')
            # row=row+1

        self.grid.AutoSize()  # 设置行和列自动调整
        f.close()
        df.close()

        h_sizer_table0 = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer_table0.Add(self.grid, 1, flag=wx.ALL, border=20)
        self.panel.SetSizer(h_sizer_table0)
        # 绑定事件
        self.Bind(wx.grid.EVT_GRID_CELL_LEFT_DCLICK, self.GetModelName)

    def GetModelName(self, event):
        row_num = event.GetRow()  # 获取点击的行序号，从零开始
        f = open(os.path.join(self.path, 'MaxMeanTrainAcc.csv'), 'r')
        f_csv = csv.reader(f)
        CData = []
        for row in f_csv:
            CData.append(row)
        f.close()
        save_name = CData[row_num + 3][1]  # save-name为双击行的模型名
        print(save_name)

        event.Skip()
