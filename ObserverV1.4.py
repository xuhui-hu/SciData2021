# coding: utf-8

import sys
import numpy as np
from skimage import transform  # 生成插值数据
import scipy.io as sio  # used for reading database
import threading
import socket
import time
from scipy import signal
import os.path
from collections import deque
# from threading import Lock, Thread
# import torch.nn as nn
# import torch

import sessantaquattro as sq  # OT HDEMG
from cyberglove import CyberGlove  # Cyberglove
import myo  # MYO Armband


from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from colour import Color
from pgcolorbar.colorlegend import ColorLegendItem
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

class RepeatedTimer(object):
    """
    A simple timer implementation that repeats itself and avoids drift over
    time.
    Implementation based on https://stackoverflow.com/a/40965385.

    Parameters
    ----------
    target : callable
        Target function
    interval : float
        Target function repetition interval
    name : str, optional (default: None)
        Thread name
    args : list
        Non keyword-argument list for target function
    kwargs : key,value mappings
        Keyword-argument dict for target function
    """

    def __init__(self, target, interval, args=(), kwargs={}):
        self.target = target
        self.interval = interval
        self.args = args
        self.kwargs = kwargs

        self._timer = None
        self._is_running = False
        self._next_call = time.time()

    def _run(self):
        self._is_running = False
        self.start()
        self.target(*self.args, **self.kwargs)

    def start(self):
        if not self._is_running:
            self._next_call += self.interval
            self._timer = threading.Timer(self._next_call - time.time(),
                                          self._run)
            self._timer.start()
            self._is_running = True

    def stop(self):
        self._timer.cancel()
        self._is_running = False


# class CNN(object):
#     def __init__(self):
#         # 加载模型的时间还是挺长的
#         print("CNN模型加载中...")
#         self.model = torch.load('model_best.pkl', map_location=torch.device('cpu'))
#         self.min_max_scaler = joblib.load('scalar01.joblib')
#         print("模型加载完毕！")
#
#     def predict(self, input):
#         # input是40*130，输入model的需要是 1,1,40,130
#         x = torch.from_numpy(np.array([[input]])).type(torch.float)
#         with torch.no_grad():
#             test_pred = self.model(x)
#             test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
#         return test_label


def messageDialog(dialogText, title='Alarm'):
    # 核心功能代码就两行，可以加到需要的地方
    if title == 'Alarm':
        msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, '警告', dialogText)
    elif title == "Prompt":
        msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, '提示', dialogText)
    elif title == "Confirm":
        msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, '确认', dialogText,
                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, )
        # msg_box.exec_()

        # return msg_box.buttonClicked(QtWidgets.QMessageBox.Ok)
    msg_box.exec_()


class ChildWin(QtWidgets.QMainWindow):
    # 定义信号
    _signal = pyqtSignal(list)

    def __init__(self):
        super(ChildWin, self).__init__()
        self.ui = uic.loadUi("Qt_UI/inputDialog.ui")
        self.ui.buttonBox.accepted.connect(self.ok)

        # self.ui.buttonBox.rejected.connect(self.cancel)

    def ok(self):
        print("ok")
        subj_id = self.ui.id.text()
        age = self.ui.age.text()
        gender = self.ui.gender.currentText()
        domain = self.ui.domain.currentText()
        tip = self.ui.tip.text()
        if len(subj_id + age) == 0:
            messageDialog("至少填写编号，请重新填写！")
        else:
            self._signal.emit([subj_id, age, gender, domain, tip])
        self.ui.id.setText("")
        self.ui.age.setText("")
        self.close()

    # def cancel(self):
    #     print("cancel")
    # data_str = self.lineEdit.text()
    # #发送信号
    # self._signal.emit(data_str)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = uic.loadUi("Qt_UI/mainV1_3.ui")
        self.ui.start.setEnabled(True)
        self.ui.stop.setEnabled(False)
        self.ui.start.clicked.connect(self.pushStart)  # 启动按键
        self.ui.stop.clicked.connect(self.pushStop)  # 停止按键
        self.ui.quit.clicked.connect(self.pushQuit)  # 退出按键
        self.ui.buttonGroup.buttonClicked.connect(self.CollectOrPredict)  # 预测采集状态切换
        # self.ui.buttonGroup_2.buttonClicked.connect(self.GuidingCurve)  # 预测采集状态切换 好像不需要专门连接到函数

        self.ui.hdemgEnable.stateChanged.connect(lambda: self.debug(self.ui.hdemgEnable))  # for debug
        self.ui.myoEnable.stateChanged.connect(lambda: self.debug(self.ui.myoEnable))  # for debug
        self.ui.gloveEnable.stateChanged.connect(lambda: self.debug(self.ui.gloveEnable))  # for debug
        self.ui.groupBox.setEnabled(True)
        self.hdemgEnable = True
        self.gloveEnable = True
        self.myoEnable = True

        self.ui.saveDataBox.setEnabled(False)
        self.ui.saveDataBox.accepted.connect(self.saveData)  # buttonGroup
        self.ui.saveDataBox.rejected.connect(self.cancelDataCollection)
        self.ui.keyPressEvent = self.keyPressEvent  # 这句要加 不然键盘按键没反应
        # self.ui.keyReleaseEvent = self.keyReleaseEvent  # 这句要加 不然键盘按键没反应
        self.ui.loadUser.clicked.connect(self.loadUser)
        '''# 用于用户数据添加的子窗口配置'''
        self.inputUser = ChildWin()  # 子窗口实例化
        self.ui.addUser.clicked.connect(self.addUser)
        # 连接信号必须放在初始化函数中，否则点击几次，下一次就会重复出现几次
        self.inputUser._signal.connect(self.getUserData)

        '''# CNN配置'''
        # self.cnn = CNN()
        '''热力图配置，之后单独写个类封装起来'''
        blue, red = Color('blue'), Color('red')
        colors = blue.range_to(red, 256)
        colors_array = np.array([np.array(color.get_rgb()) * 255 for color in colors])
        look_up_table = colors_array.astype(np.uint8)
        self.heatImage = pg.ImageItem()
        self.heatImage.setOpts(axisOrder='row-major')  # 2021/01/19 Add
        self.heatImage.setLookupTable(look_up_table)
        self.heatImage.setImage(transform.resize(np.random.rand(5, 13), (500, 1300)))
        view_box = pg.ViewBox()
        view_box.setAspectLocked(lock=True)
        view_box.addItem(self.heatImage)
        plot = pg.PlotItem(viewBox=view_box)
        color_bar = ColorLegendItem(imageItem=self.heatImage, showHistogram=True, label='RMS (mV)')
        # color_bar.resetColorLevels()  #这条会报错 说找不到属性
        self.ui.heatMap.addItem(plot)
        self.ui.heatMap.addItem(color_bar)
        plot.setMouseEnabled(x=False, y=False)
        plot.setRange(xRange=[0, 1300], yRange=[0, 500], padding=0)
        '''左右坐标（共用横坐标）'''
        self.cyberglove = self.ui.cyberglove.addPlot()  # A plot area (ViewBox + axes) for displaying the image

        self.gloveInterval = 0.05  # 数据手套读取间隔
        self.historyLength = int(self.ui.SecondPerTrial.value() / self.gloveInterval)  # 横坐标长度
        self.cyberglove.showGrid(x=True, y=True)  # 把X和Y的表格打开
        self.cyberglove.setRange(xRange=[0, self.historyLength], yRange=[-50, 110], padding=0)

        self.cyberglove.setLabel('bottom', 'Time', units='Seconds', **{'font-size': '10pt'})
        self.cyberglove.getAxis('bottom').setPen(pg.mkPen(color='#000000', width=3))  # 横坐标

        self.cyberglove.setLabel('left', 'Glove', units='Degree', color='#c4380d', **{'font-size': '10pt'})
        self.cyberglove.getAxis('left').setPen(pg.mkPen(color='#c4380d', width=3))  # 左侧纵坐标

        self.cyberglove.setLabel('right', 'Guiding Curve', units="Degree", color='#025b94', **{'font-size': '10pt'})
        self.cyberglove.getAxis('right').setPen(pg.mkPen(color='#025b94', width=3))  # 右侧纵坐标
        self.cyberglove.setMouseEnabled(x=False, y=True)

        self.targetCurve = pg.ViewBox()
        self.cyberglove.scene().addItem(self.targetCurve)
        self.cyberglove.getAxis('right').linkToView(self.targetCurve)
        self.targetCurve.setXLink(self.cyberglove)
        self.targetCurve.setYRange(10, 90)
        self.targetCurve.setMouseEnabled(x=False, y=False)
        self.gloveSeq = []
        # self.emgRmsSeq = []

        self.glovePlot = self.cyberglove.plot(pen=pg.mkPen(color='#c4380d', width=3))
        # self.emgRmsPlot = pg.PlotCurveItem(
        #     pen=pg.mkPen(color='#025b94', width=2, style=QtCore.Qt.DotLine))  # 2021.6.12 现不加入EMG
        self.curvePlot = pg.PlotCurveItem(pen='w')  # 蓝色曲线
        self.targetCurve.addItem(self.curvePlot)
        # self.emgRMS.addItem(self.emgRmsPlot)

        self.updateViews()
        self.cyberglove.getViewBox().sigResized.connect(self.updateViews)

        # 启动定时器，约60Hz刷新率，刷新数据显示
        self.ui.timer = QtCore.QTimer()
        self.ui.timer.timeout.connect(self.updateData)
        self.ui.timer.start(20)  # 20ms刷新周期 50Hz

        self.collectStop = True  # True：不显示采集的数据 False：显示采集的数据
        self.stop = True  # True：停止采集 False：启动采集

        '''HDEMG采集相关'''
        self.arrayIndex = np.loadtxt("sessantaquattro/GR08MM1305.txt", dtype=int)  # 电极编号
        self.status = self.ui.buttonGroup.checkedButton().objectName()  # 默认初始模式为'predict' 可切换到采集模式
        self.online = True
        self.monopole = True  # 降噪性能一般，并容易使激活区域失真

        # Create start command and get basic setup information
        (self.command, self.number_of_channels,
         self.sample_frequency, self.bytes_in_sample) = sq.create_bin_command(mode=0)
        # mode = 7: test mode;mode = 0:Monopolar mode
        self.connection = None

        self.rawEMG = np.zeros((self.number_of_channels - 4,))
        # 2021.8.10 这里因为要组成5x13的图绘制在上位机，因此总共65个点
        self.envelopEMG = np.zeros((self.number_of_channels - 3,))
        '''HD-sEMG 提取包络信号'''
        # 常规的操作是先对原始信号做50Hz高通滤波，这有助于去除直流偏置，然后再整流 做低通，这里直接做10Hz的三阶低通（原本是四阶，这里考虑减小计算量）
        # 详见 https://c-motion.com/v3dwiki/index.php?title=EMG_Linear_Envelope
        cutoffFreq = 10
        self.b, self.a = signal.butter(N=4, Wn=2 * cutoffFreq / self.sample_frequency, btype='lowpass', analog=False)
        self.zi = signal.lfilter_zi(self.b, self.a) * np.ones((self.rawEMG.shape[0], 1))  # 同时对所有通道滤波，初始状态包括64个通道
        # print(self.zi.shape)

        '''数据储存变量'''
        self.trialSize = self.ui.SecondPerTrial.value() * self.sample_frequency
        self.trialData = []
        self.gIndexStart = 1
        self.gNum = 12  # 手势种类
        self.dataSize = np.zeros((self.gNum,), dtype=int)  # 记录各个手势的trial次数

        self.dataStatus = []  # 记录是否开始采集，作为按下空格的标志位
        for i in range(self.gNum):
            self.dataStatus.append(False)  # 存储每个数据的采集状态，按下空格，置位True
        self.histDataStatus = []  # 记录历史采集状态，主要在UpdateData里使用
        for i in range(self.gNum):
            self.histDataStatus.append(False)

        self.gIndex = self.gIndexStart  # 0开始，0~12  # 标记需要记录的手势的编号
        self.trialNum = 0  # 在一次trial中采集的样本数计数，dataSize 是用来计trial次数的
        self.endtrialFLag = False

        '''手套相关'''
        self.gloveNum = 15  # 4-18(0开头) 共15通道
        self.gloveData = np.zeros((self.gloveNum,))
        self.gloveVisualData = np.zeros((6,))
        # 元素存储的是gloveVisualData的下标，共12个，每个手势取对应的gloveVisualData作为连续标签
        # 小拇指运动幅度小，复合运动中，能用其他手指代替的就用其他手指
        self.gloveVisualDataList = [0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 4, 5, 0, 1, 3, 0, 1, 0]
        self.gloveVisualDataIdx = 0
        self.addr = None
        self.cg = None
        self.gloveThread = None

        # 手势图和标签初始化
        getattr(self.ui, 'g' + str(self.gIndex).zfill(2)).setStyleSheet("background-color:yellow")  # 初始第一个label为黄色
        self.ui.prediction.setPixmap((QPixmap('Qt_UI/Fig/%s.png' % self.gIndex)))
        self.ui.prediction.setScaledContents(True)  # 图像在pixmap中的尺寸自适应
        '''
        加载用户数据列表
        '''
        if not os.path.exists('./DataSet'):
            os.mkdir('./DataSet')
            print("未发现数据集路径，已新建")
        else:
            if os.path.exists('./DataSet/Info.txt'):
                with open("./DataSet/Info.txt", "r") as f:
                    data = f.readlines()
                for i in data:
                    userList = i.split(' ')[0] + " " + i.split(' ')[1]
                    self.ui.userList.addItem(userList)

        '''MYO配置'''
        myo.init()
        self.hub = myo.Hub()
        self.listener = myo.EmgCollector(2000)  # 200Hz采样率，x轴长度为2000
        self.myoThread = None

        '''MYO窗口'''
        self.myoChannel = []
        for i in range(8):
            ch = self.ui.myo.addPlot()
            ch.setRange(xRange=[0, self.listener.n], yRange=[-100, 100], padding=0)
            self.myoChannel.append(ch.plot(pen=pg.mkPen(color='#c4380d', width=1)))
            if (i + 1) % 2 == 0:
                self.ui.myo.nextRow()

        '''其他初始化'''
        self.gloveFpsCount = 0
        self.emgFpsCount = 0
        self.gloveT0 = 0
        self.gloveT1 = 0
        self.emgT0 = 0
        self.emgT1 = 0
        self.autoSpeedSetting()  # 自动选择低速或慢速
        self.trialNum = 0  # 采集样本数计数
        self.addr = None  # 文件存储路径

    def updateViews(self):
        self.targetCurve.setGeometry(self.cyberglove.getViewBox().sceneBoundingRect())
        self.targetCurve.linkedViewChanged(self.cyberglove.getViewBox(), self.targetCurve.XAxis)

    def pushStart(self):  # 所有数据的总开关，需要先加载用户数据才能
        if self.addr is None:  # 未指定路径
            messageDialog("请先加载对象！")
            return
        self.ui.start.setEnabled(False)
        self.ui.stop.setEnabled(True)
        self.ui.groupBox.setEnabled(False)  # 开始后不允许调试
        self.stop = False
        self.collectStop = False
        if self.online:
            if self.hdemgEnable:
                '''启动HDEMG'''
                ip_address = '0.0.0.0'
                port = 45454
                # Create a socket which is used to connect to Sessantaquattro
                sq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sq_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sq_socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

                print('Starting to log data: {0} channels, {1} Hz sampling rate, {2} bits ADC precision'.format(
                    self.number_of_channels,
                    self.sample_frequency,
                    self.bytes_in_sample * 8))
                self.ui.emgFPS.display(self.sample_frequency)  # 显示sq的采样率

                # Open connection to Sessantaquattro
                self.connection = sq.connect_to_sq(sq_socket, ip_address, port, self.command)

                thread = threading.Thread(target=self.recordThread)  # 启动HDEMG采集线程
                thread.start()

            if self.gloveEnable:
                '''启动CyberGlove采集线程'''
                user = self.ui.userList.currentText()
                self.addr = os.path.join('DataSet', user.split(" ")[0])  # 添加到对象一级的路径
                self.cg = CyberGlove(n_df=22, s_port='COM' + str(self.ui.COMPort.value()),
                                     samples_per_read=1, cal_path=os.path.join(self.addr, user.split(" ")[0] + '.cal'))
                self.gloveThread = RepeatedTimer(self.gloveReading, self.gloveInterval, kwargs={"cyberglove": self.cg})
                self.ui.gloveFPS.display(int(1 / self.gloveInterval))

                self.gloveThread.start()

            if self.myoEnable:
                '''启动MYO'''
                self.ui.myoFPS.display(200)  # int(self.listener.rate) # 采样率经实测稳定在200Hz左右
                self.myoThread = threading.Thread(target=lambda: self.hub.run_forever(self.listener.on_event, 500))
                self.myoThread.start()
                print("start record!")
        else:
            print("start loading offline data!")

    def pushStop(self):
        self.ui.start.setEnabled(True)
        self.ui.stop.setEnabled(False)
        self.ui.groupBox.setEnabled(True)  # 结束后才允许调试
        self.stop = True
        if self.online:
            if self.gloveEnable:
                self.gloveThread.stop()
                time.sleep(0.5)
                self.cg.stop()  # 模板程序中 在停止时，两个都停止了，因此如果重新按start，两个都得再启动。否则可以在关闭窗口时再执行cg.stop,未测试。
            if self.myoEnable:
                self.hub.stop()
            # self.myoThread.stop()
            print("stop record!")
        else:
            print("stop loading offline data!")

    def pushQuit(self):
        pass

    def CollectOrPredict(self):
        self.status = self.ui.buttonGroup.checkedButton().objectName()
        if self.status == 'predict':
            self.ui.SecondPerTrial.setEnabled(False)
        else:
            self.ui.SecondPerTrial.setEnabled(True)

    def addUser(self):  # 打开录入用户信息的子窗口
        print('打开子窗口！')
        self.inputUser.ui.show()

    def loadUser(self):
        user = self.ui.userList.currentText()
        self.addr = os.path.join('DataSet', user.split(" ")[0])  # 添加到对象一级的路径
        for i in range(self.gIndexStart, self.gNum + 1):
            subAddr = os.path.join(self.addr, str(i).zfill(2) + '.txt')  # 添加到手势一级的路径
            if not os.path.exists(subAddr):  # 如果文件不存在
                self.dataSize[i - 1] = 0
            else:
                print("正在加载第%s个手势的数据，请稍候" % i)
                # tp = np.loadtxt(subAddr)
                count = 0
                with open(subAddr) as f:
                    while True:
                        buffer = f.read(1024*8192)
                        if not buffer:
                            break
                        count += buffer.count('\n')
                # 数据集的样本数赋值给相应位置的dataSize
                self.dataSize[i - 1] = count / (self.sample_frequency * self.ui.SecondPerTrial.value())  #
            getattr(self.ui, 's' + str(i).zfill(2)).setText(str(self.dataSize[i - 1]))  # 先获取到.ui的label对象，然后修改值

        print("加载完毕！")
        self.ui.dataStatus.setText("数据加载成功！：" + user)

    def getUserData(self, parameter):  # 添加用户信息
        Dataset_dir = os.listdir('DataSet')
        if not Dataset_dir:
            idx = 's00'
        else:  # 如果存有文件，读取info的最后一行
            with open("./DataSet/Info.txt", "r") as f:
                data = f.readlines()
            # 取最后一行，按空格split，取第0个元素的第一个字符到最后
            idx = 's' + str(int(data[-1].split(" ")[0][1:]) + 1).zfill(2)
        # 创建存放数据的路径
        parameter.insert(0, idx)

        os.mkdir(os.path.join('./DataSet', idx))
        # 写Info文件
        with open("./DataSet/Info.txt", "a+") as f:
            for i in range(len(parameter)):
                f.write(str(parameter[i]) + " ")
            f.write("\n")

        self.ui.userList.addItem(parameter[0] + " " + parameter[1])
        # print("添加用户成功:" + str(parameter[1]))
        messageDialog("添加用户成功:" + str(parameter[1]), 'Prompt')

    def recordThread(self):  # HD-sEMG的数据采集线程
        while not self.stop:
            '''HDEMG 采集'''
            self.getHDEMG()
            self.dataCollection()  # 数据采集
        sq.disconnect_from_sq(self.connection, self.command)

    def getHDEMG(self):  # read HD-sEMG of sq, and calculate envelope of HD-sEMG
        sample_from_channels = sq.read_data(self.connection, self.number_of_channels, self.bytes_in_sample)  # 一帧数据
        # print(sample_from_channels)
        self.rawEMG = sample_from_channels[0][0:-4]
        # np.array(sample_from_channels)
        self.envelopEMG = self.linearEnvelope()

    def dataCollection(self):  # 采集数据的文件写入
        """
        先存在数组里，如果数据质量符合标准，那么在确认后再保存。
        """
        if self.dataStatus[self.gIndex - 1]:  # 如果选中的手势被触发了
            # self.trialData[self.trialNum, :] = np.hstack((self.rawEMG, self.gloveData))  # 存储数据
            self.trialData.append(np.hstack((np.array(self.rawEMG).reshape((-1,)),
                                             self.gloveData,
                                             np.array(self.listener.emg).reshape((8,)),
                                             np.array(self.listener.euler).reshape((3,))
                                             )))
            self.trialNum += 1
            # if self.endtrialFLag:  # 每个trail的样本数为采样率x采样时间
            if self.trialNum >= self.trialSize:  # 总样本数
                self.dataStatus[self.gIndex - 1] = False  # 停止采集
                self.trialNum = 0  # 计数清0
                self.collectStop = True  # 只关闭显示，不关闭采集 self.pushStop()
                self.ui.saveDataBox.setEnabled(True)  # 使能保存按钮
                print("一轮采集结束！,确认是否保存？ 保存请按")
                getattr(self.ui, 'g' + str(self.gIndex).zfill(2)).setStyleSheet(
                    "background-color:yellow;color:black")  # 松开空格，label变黑
                getattr(self.ui, 's' + str(self.gIndex).zfill(2)).setStyleSheet("color:black")  # 松开空格，数字变黑

                # self.endtrialFLag = False

            # with open("./Data/%s.txt" % str(self.gIndex).zfill(2), "a+") as file:
            #     np.savetxt(file, self.rms.reshape(1, -1))

    def updateData(self):  # 刷新上位机数据
        # print(self.stop,self.collectStop)
        if (not self.stop) and (not self.collectStop):
            '''刷新热力图数据'''
            if self.monopole:  # 如果是单端采集
                array_rms = np.zeros((5, 13))
                for i in range(self.arrayIndex.shape[0]):
                    for j in range(self.arrayIndex.shape[1]):
                        # print(self.arrayIndex[i, j], self.rms.shape)
                        array_rms[i, j] = self.envelopEMG[self.arrayIndex[i, j]]
                array_rms = np.flip(array_rms, axis=0)  # 不知道为什么画图的时候，数据被翻转了
                data = transform.resize(array_rms, (500, 1300))  # order参数选择插值方法，默认是bi-linear
                self.heatImage.setImage(data)
            else:  # 如果是双端采集 维度为4行13列
                array_rms = np.flip(self.rms.reshape(4, 13), axis=0)  # 不知道为什么画图的时候，数据被翻转了
                data = transform.resize(array_rms, (500, 1300))  # order参数选择插值方法，默认是bi-linear
                self.heatImage.setImage(data)

            '''刷新手套 窗口为10s'''  # 填充时间和该函数的刷新率有关
            self.glovePlot.setData(self.gloveSeq)

            '''刷新MYO'''
            emg_data = self.listener.get_emg_data()
            emg_data = np.array([x[1] for x in emg_data]).T
            for g, data in zip(self.myoChannel, emg_data):
                g.setData(list(data))

            # print(len(self.listener.emg_data_queue))

            '''清除序列条件'''
            if self.dataStatus[self.gIndex - 1]:  # 如果选中的手势触发了"数据采集指令"
                if self.histDataStatus[self.gIndex - 1] != self.dataStatus[self.gIndex - 1]:  # 上一次没触发 说明刚进入
                    self.gloveSeq = []
                    self.listener.emg_data_queue = deque(maxlen=self.listener.n)  # MYO序列也清除
            else:  # 如果没有触发“采集指令”， 那么到达窗口最大范围时自动清0
                # print(len(self.listener.emg_data_queue))
                if len(self.listener.emg_data_queue) >= self.listener.n:
                    self.gloveSeq = []
                    self.listener.emg_data_queue = deque(maxlen=self.listener.n)  # MYO序列也清除
            self.histDataStatus[self.gIndex - 1] = self.dataStatus[self.gIndex - 1]
            '''刷新手势预测值'''
            # if self.status == 'predict':
            #     cnn_input = self.cnn.min_max_scaler.transform(self.rms.reshape(1, -1))
            #     gesture = self.cnn.predict(transform.resize(cnn_input.reshape(4, 13), (40, 130)))
            #     self.ui.prediction.setPixmap((QPixmap('Fig/%s.png' % gesture[0])))
            #     self.ui.prediction.setScaledContents(True)

        '''调节运动的标定轨迹'''
        if self.ui.buttonGroup_2.checkedButton().objectName() == 'lowSpeed':
            emgMax = self.ui.maxSlider.value()
            emgMin = self.ui.miniSlider.value()
            curve = np.zeros((self.historyLength,))
            curve[0:10] = np.ones((10,)) * emgMin
            if emgMax - emgMin == 0:
                curve[10:70] = np.ones((60,)) * emgMax
                curve[130:190] = np.ones((60,)) * emgMax
            else:
                curve[10:70] = np.arange(emgMin, emgMax, (emgMax - emgMin) / 60)
                curve[130:190] = np.arange(emgMax, emgMin, (emgMin - emgMax) / 60)
            curve[70:130] = np.ones((60,)) * emgMax
            curve[190:] = np.ones((10,)) * emgMin

            self.curvePlot.setData(curve, pen='w')
            # self.ui.test.setText(str(self.gIndex))
        else:
            emgMax = self.ui.maxSlider.value()
            emgMin = self.ui.miniSlider.value()
            curve = np.zeros((self.historyLength,))
            curve[0:40] = np.ones((40,)) * emgMin
            if emgMax - emgMin == 0:
                curve[40:60] = np.ones((20,)) * emgMax
                curve[140:160] = np.ones((20,)) * emgMax
            else:
                curve[40:60] = np.arange(emgMin, emgMax, (emgMax - emgMin) / 20)
                curve[140:160] = np.arange(emgMax, emgMin, (emgMin - emgMax) / 20)
            curve[60:140] = np.ones((80,)) * emgMax
            curve[160:] = np.ones((40,)) * emgMin

            self.curvePlot.setData(curve, pen='w')

    def gloveReading(self, cyberglove):  # 存储原始数据至数据集，处理关节角用于可视化
        raw_data = cyberglove.read()
        '''存入全局变量'''
        self.gloveVisualData[0] = raw_data[4]  # 食指
        self.gloveVisualData[1] = raw_data[7]  # 中指
        self.gloveVisualData[2] = raw_data[11]  # 无名指
        self.gloveVisualData[3] = raw_data[15]  # 小拇指
        self.gloveVisualData[4] = 60 - raw_data[18]  # 手指外展
        self.gloveVisualData[5] = raw_data[18]  # 手指内收，直接利用自动对齐纵坐标

        self.gloveData = raw_data[4:19].reshape((-1,))

        self.gloveSeq.append(self.gloveVisualData[self.gloveVisualDataIdx])  # 0.05s刷新，200计满10s
        # print(self.gloveVisualData)

    def keyPressEvent(self, e):
        if self.status != 'collect':
            messageDialog('请先切换至采集模式！')
            return
        if e.isAutoRepeat():  # 检测事件是否重复触发，因此只响应一次
            pass
        else:  # 使用elif，一次只响应一个键
            if e.key() == Qt.Key_Up:  # 可能需要定义一个字典
                # print("up")
                self.gIndex -= 1
                if self.gIndex < self.gIndexStart:
                    self.gIndex = self.gNum
                for i in range(self.gIndexStart, self.gNum + 1):
                    getattr(self.ui, 'g' + str(i).zfill(2)).setStyleSheet(
                        "background-color:transparent")  # 先获取到.ui的label对象，然后修改值
                getattr(self.ui, 'g' + str(self.gIndex).zfill(2)).setStyleSheet("background-color:yellow")
                self.ui.prediction.setPixmap((QPixmap('Qt_UI/Fig/%s.png' % self.gIndex)))
                self.ui.prediction.setScaledContents(True)
                self.gloveVisualDataIdx = self.gloveVisualDataList[self.gIndex - 1]  # 设置好每个手势对应读取的gloveVisualData的下标
            elif e.key() == Qt.Key_Down:
                # print("down")
                self.gIndex += 1
                if self.gIndex > self.gNum:
                    self.gIndex = self.gIndexStart
                for i in range(self.gIndexStart, self.gNum + 1):
                    getattr(self.ui, 'g' + str(i).zfill(2)).setStyleSheet(
                        "background-color:transparent")  # 先获取到.ui的label对象，然后修改值
                getattr(self.ui, 'g' + str(self.gIndex).zfill(2)).setStyleSheet("background-color:yellow")
                self.ui.prediction.setPixmap((QPixmap('Qt_UI/Fig/%s.png' % self.gIndex)))
                self.ui.prediction.setScaledContents(True)
                self.gloveVisualDataIdx = self.gloveVisualDataList[self.gIndex - 1]  # 设置好每个手势对应读取的gloveVisualData的下标
            elif e.key() == Qt.Key_Space:
                print("开始采集！")
                self.dataStatus[self.gIndex - 1] = True  # 对选中的手势触发“采集”指令
                # self.trialNum = 0  # 采集trial的计数变量
                getattr(self.ui, 'g' + str(self.gIndex).zfill(2)).setStyleSheet(
                    "background-color:yellow;color:red")  # label变红
                getattr(self.ui, 's' + str(self.gIndex).zfill(2)).setStyleSheet("color:red")  # 数字变红

                # 开始采集时，清空图中的曲线
                self.gloveSeq = []
                self.listener.emg_data_queue = deque(maxlen=self.listener.n)  # MYO序列也清除
                self.trialData = []  # 清空数据内存
                self.trialNum = 0  # 清空计数

            self.autoSpeedSetting()

    def linearEnvelope(self):  # 用于计算heatMap，该函数生成每个HD-sEMG的通道的包络信号

        envelopEMG, self.zi = signal.lfilter(self.b, self.a,
                                             np.abs(self.rawEMG).reshape((-1, 1)),
                                             zi=self.zi, axis=1)
        # print("line652:", end="")
        # print(self.rawEMG.shape, np.array(envelopEMG).shape, self.zi.shape)
        return np.vstack(([0], envelopEMG))

    def saveData(self):  # 采集完一轮后，按下确认按钮则保存
        with open(os.path.join(self.addr, "%s.txt" % str(self.gIndex).zfill(2)), "a+") as file:
            # 设一次采集10s，理论上应该是20000（2000 * 10）个数据，由于采集的数量有一定误差，会超过10000，因此掐掉了头尾，截取至10000。
            # dataLen = self.sample_frequency * self.ui.SecondPerTrial.value()
            if len(self.trialData) - self.trialSize == 0:
                np.savetxt(file, np.array(self.trialData))
                self.dataSize[self.gIndex - 1] += 1
                getattr(self.ui, 's' + str(self.gIndex).zfill(2)).setText(str(self.dataSize[self.gIndex - 1]))
            else:
                print("实际保存的数据长度(%s)与预期(%s)不符,数据保存中止" % (len(self.trialData), self.trialSize))
            self.trialData = []

        print("保存完毕!")
        self.autoSpeedSetting()  # 更新下一次的配速
        self.ui.saveDataBox.setEnabled(False)
        self.collectStop = False  # 数据保存完后，继续显示

    def cancelDataCollection(self):  # 采集完一轮后，按下确认按钮则保存
        print("取消本轮数据保存!")
        self.ui.saveDataBox.setEnabled(False)
        self.collectStop = False  # 数据保存完后，继续显示

    def autoSpeedSetting(self):  # 设定速度 刚启动时、切换时、保存时均检查 每6轮一个周期
        if int(getattr(self.ui, 's' + str(self.gIndex).zfill(2)).text()) % 6 < 3:
            self.ui.lowSpeed.setChecked(True)
        else:
            self.ui.highSpeed.setChecked(True)

    def debug(self, chbox):
        if chbox.isChecked():
            if chbox is self.ui.hdemgEnable:
                self.hdemgEnable = True
                print("启用HDEMG")
            elif chbox is self.ui.gloveEnable:
                self.gloveEnable = True
                print("启用手套")
            elif chbox is self.ui.myoEnable:
                self.myoEnable = True
                print("启用MYO")
        else:
            if chbox is self.ui.hdemgEnable:
                self.hdemgEnable = False
                print("禁用HDEMG")
            elif chbox is self.ui.gloveEnable:
                self.gloveEnable = False
                print("禁用手套")
            elif chbox is self.ui.myoEnable:
                self.myoEnable = False
                print("禁用MYO")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.ui.show()
    sys.exit(app.exec_())
