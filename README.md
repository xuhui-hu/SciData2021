# SIEM Dataset
 Simultaneous recordings of Intrinsic and Extrinsic Muscles using HD-sEMG  for dynamic finger movement decoding

 * The software is written by Python 3.8 and compatible with Python 3.6
 * UI is designed by PyQt5, 
### Instruction
The collected gestures are as follows:

| 1 finger | 2 and 3 fingers      | 4 fingers and others   |  
|  ----    | ----                 |       ----             |  
|01 index  |05 index+middle       |  10 fingers flex       |                    
|02 middle |06 middle+ring        |  11 fingers abduction  |               
|03 ring   |07 ring+little        |   12 fingers adduction |                            
|04 little |08 index+middle+ring  |                        |
|          |09 middle+ring+little |                        |

### Functions
1. 添加和加载用户
2. 一次存储10s数据，期间按空格可以重新采集
3. 可视化包括HD-sEMG MYO Cyberglove 三种数据
4. 后期加入手势识别程序

### Notes：
1. 目前设 self.online = True, self.monopole = True
2. 用low-pass filter 计算emg的envelop。为了获取heatMap，编号重布局在60Hz刷新率的updateData里完成
3. 在recordThread中，只做两件事，存self.rawEMG和self.envelopEMG
4. 读取的HD-sEMG的有效通道数为64，第65个是取平均得到的，因此数据集中HD-sEMG的通道数依然为64

### History Version
2021.8.12
1. 优化为英文界面
2. 优化读取采集次数的速度
3. 优化上位机图像
4. 修复之前按下空格但没有重启采集的问题
5. 修复只能对6轮采集进行速度自动设定的问题

2021.8.10
1. 同步数据存取和数据可视化的时刻
2. 开始前检查是否加载用户

2021.8.9
修复之前程序可能出现的数据采集长度有长有短的问题。

2021.8.8
优化了程序结构，并放在github托管
去掉了self.rms和self.datawindow,原本用rms来算，现在直接用低通滤波获取

2021.8.4 可见我鸽了多久，，
1. 检查程序逻辑，增加imu
2. OT设备改为2000Hz采样率 重新写sq的通信协议

2021.6.18
1. 解决数据不满10000的问题,两边平均用0补齐
2. 以及在采集过程中，再次按空格重新采数据的问题
3. 把MYO的窗口也设为10s长的
4. 快速再快一点,慢速再慢一点,之前差异比较小,并且固定标定曲线,只能拖动数据手套的轴.
5. 修复了一个bug，之前存储数据手套数据的时候，漏存了最后一个小拇指伸展的数据。

2021.6.15
1. 确认保存数据后，不需要再start 节约训练时间
2. 自动高速低速切换，也可以手动切换。切换需要采集的手势时，读取采集次数，确定本轮需要的配速。另外在一次采集结束后，也增加一个这样的判断
3. 数据手套的标定，能不使用小拇指，就不使用

2021.6.11 V1.3 
1. 增加MYO
2. 增加伸指手势
3. 采用数据手套标定运动 就从Dynamic运动入手 

2021.6.1 V1.2.1
1. 去掉放松手势的采集,调整为连续手势采集（定时采数据，采集时间结束，弹窗确认是否保存。）
2. 增加数据手套的接口，可视化界面分为上下两张图：
    上部的图包含一个理想的、位置轨迹。同时有数据手套的轨迹，以规范关节角速度
    下部的图是HDEMG所有通道的平均值，用于可视化force信息
3. 单端采集，差分显示。数据集使用单端数据，离线或在线处理时使用差分处理
4. 加入数据采样率显示

2021.5.28 V1.2 
1. 采用KNN做预测，采集多个对象的，多轮采集
2. 制定一个数据采集协议，为之后数据开源做准备。
3. 每个受试者的完成手势的顺序：
4. 采集手指连续运动标签。（上位机增加手指运动标签的可视化）

2021.5.26  V1.1
1. 由于新增了大量的手势 这里再用手势命名太麻烦 现在全部改为数字命名
2. 增加了频域分析的图，观察工频干扰情况 在fftPlot.py。注意输入fft的序列长度数值至少为采样频率N， FFT反应原始频谱，
3. 剩下要做的，包括先去工频干扰，再做信号差分