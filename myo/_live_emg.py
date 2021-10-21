# The MIT License (MIT)
#
# Copyright (c) 2017 Niklas Rosenstein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

from matplotlib import pyplot as plt
from collections import deque
# from threading import Lock, Thread
import threading
import myo
import numpy as np
import time
import sys
import collections
import math


class EmgCollector(myo.DeviceListener):
    """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

    def __init__(self, n):
        super(EmgCollector, self).__init__()

        self.n = n
        self.lock = threading.Lock()
        self.emg_data_queue = deque(maxlen=n)
        self.times = deque()
        self.last_time = None
        self.emg = 0
        # IMU
        # self.imu_data_queue = deque(maxlen=n)
        self.orientation = None
        self.euler = None

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    # myo.DeviceListener
    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))
            self.emg = event.emg

            # # 获取采样频率
            # t = time.perf_counter()
            # if self.last_time is not None:
            #     self.times.append(t - self.last_time)
            #     if len(self.times) > self.n:
            #         self.times.popleft()
            # self.last_time = t

    def on_orientation(self, event):
        with self.lock:
            self.orientation = event.orientation
            # 这边做一个校准，可能要加一个IMU的校准按钮
            x, y, z, w = self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w
            roll = math.atan2(2 * w * x + 2 * y * z, 1 - 2 * x * x - 2 * y * y) * 180 / 3.1415926
            pitch = math.asin(max(-1, min(1, 2 * w * y - 2 * z * x))) * 180 / 3.1415926
            yaw = math.atan2(2 * w * z + 2 * x * y, 1 - 2 * y * y - 2 * z * z) * 180 / 3.1415926
            self.euler = (roll, pitch, yaw)

            # self.imu_data_queue.append((event.timestamp, self.euler))
            # self.output()
    @property
    def rate(self):
        if not self.times:
            return 0.0
        else:
            return 1.0 / (sum(self.times) / float(self.n))


class Plot(object):
    def __init__(self, listener):
        self.n = listener.n
        self.listener = listener
        self.fig = plt.figure()
        self.axes = [self.fig.add_subplot('42' + str(i)) for i in range(1, 9)]
        [(ax.set_ylim([-100, 100])) for ax in self.axes]
        self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
        plt.ion()

    def update_plot(self):
        emg_data = self.listener.get_emg_data()
        emg_data = np.array([x[1] for x in emg_data]).T
        for g, data in zip(self.graphs, emg_data):
            if len(data) < self.n:
                # Fill the left side with zeroes.
                data = np.concatenate([np.zeros(self.n - len(data)), data])
            g.set_ydata(data)
        plt.draw()

    def main(self):
        while True:
            self.update_plot()
            plt.pause(1.0 / 30)


def main():
    myo.init(sdk_path="myo-sdk-win-0.9.0")
    hub = myo.Hub()
    listener = EmgCollector(100)
    with hub.run_in_background(listener.on_event):
        Plot(listener).main()


if __name__ == '__main__':
    main()
