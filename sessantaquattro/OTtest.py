import communication
import socket
import time
import numpy as np

ip_address = '0.0.0.0'
port = 45454

# Create a socket which is used to connect to Sessantaquattro
sq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sq_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sq_socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

# Create start command and get basic setup information

# 不支持同时TCP和SD卡 mode = 7 test模式
(command,
 number_of_channels,
 sample_frequency,
 bytes_in_sample) = communication.create_bin_command(fsamp=2, nch=3, mode=7, hres=0, hpf=1, trig=0)

print('Starting to log data: {0} channels, {1} Hz sampling rate, {2} bits ADC precision'.format(number_of_channels,
                                                                                                sample_frequency,
                                                                                                bytes_in_sample * 8))
# Open connection to Sessantaquattro
connection = communication.connect_to_sq(sq_socket, ip_address, port, command, go=1)
# start_t0 = time.time()
interval = 10  # 记录时长（second）
maxBuffSize = 1152  # 一次最大接收长度 1152 bytes

# batch = int(maxBuffSize / (number_of_channels * bytes_in_sample))  # 一次接收的最大字节数为1152
batch = 1
print("batch size : %s. Starting Collecting。。。" % batch)
totalSample = sample_frequency * interval
DATA = np.zeros((totalSample, number_of_channels))

t0 = time.time()
for i in range(int(totalSample / batch)):
    dataTemp = communication.read_data(connection, number_of_channels, bytes_in_sample, batch)
    DATA[i * batch:(i+1) * batch, :] = dataTemp
    # if i == 0:
        # print(time.time()-start_t0)  # 计算启动延时

lastBatch = totalSample % batch
print("lastBatch: %s" % lastBatch)
if lastBatch > 0:
    dataTemp = communication.read_data(connection, number_of_channels, bytes_in_sample, lastBatch)
    DATA[-lastBatch:, :] = dataTemp
print("Data Intervals: %s s. Reading Time: %s s" % (interval, time.time()-t0))
# np.savetxt("data.txt", DATA)

communication.disconnect_from_sq(connection, command)
'''
以下是12ch+500Hz+16bits的读取速度
batch size : 1. Starting Collecting。。。
Data Intervals: 10 s. Reading Time: 12.433531284332275 s
Data Intervals: 10 s. Reading Time: 12.480372428894043 s

batch size : 48. Starting Collecting。。。
Data Intervals: 10 s. Reading Time: 12.646428108215332 s
Data Intervals: 10 s. Reading Time: 13.063981294631958 s

以下是68ch+2000Hz+16bits的读取速度
batch size : 1. Starting Collecting。。。
Data Intervals: 10 s. Reading Time: 12.60048508644104 s
Data Intervals: 10 s. Reading Time: 12.33555793762207 s
Data Intervals: 10 s. Reading Time: 12.966288089752197 s

batch size : 8. Starting Collecting。。。
Data Intervals: 10 s. Reading Time: 13.121055126190186 s
Data Intervals: 10 s. Reading Time: 13.026947498321533 s
Data Intervals: 10 s. Reading Time: 12.138456583023071 s
Data Intervals: 10 s. Reading Time: 12.386715412139893 s

2021.8.8 测试下来批量和不批量好像也没什么差别 可能读取长字节的数据比多读几次花的时间更长。
'''