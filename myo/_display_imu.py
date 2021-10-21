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
"""
This example displays the orientation, pose and RSSI as well as EMG data
if it is enabled and whether the device is locked or unlocked in the
terminal.

Enable EMG streaming with double tap and disable it with finger spread.
"""

from __future__ import print_function
from myo.utils import TimeInterval
import myo
import sys
import time
from threading import Lock, Thread
import math

from matplotlib import pyplot as plt
import numpy as np
from collections import deque


class Listener(myo.DeviceListener):

    def __init__(self, n):
        self.n = n
        self.interval = TimeInterval(None, 0.05)
        self.orientation = None
        self.euler = None
        self.pose = myo.Pose.rest
        self.emg_enabled = True
        self.locked = False
        self.rssi = None
        self.emg = None
        self.lock = Lock()

        self.imu_data_queue = deque(maxlen=n)
        self.times = deque()
        # print("---")

    def get_imu_data(self):
        with self.lock:
            return list(self.imu_data_queue)

    def output(self):
        if not self.interval.check_and_reset():
            return

        parts = []
        if self.orientation:
            for comp in self.euler:
                parts.append('{}{:.3f}'.format(' ' if comp >= 0 else '', comp))
        parts.append(str(self.pose).ljust(10))
        parts.append('E' if self.emg_enabled else ' ')
        parts.append('L' if self.locked else ' ')
        parts.append(self.rssi or 'NORSSI')
        if self.emg:
            for comp in self.emg:
                parts.append(str(comp).ljust(5))
        print('\r' + ''.join('[{}]'.format(p) for p in parts), end='')
        sys.stdout.flush()

    def on_connected(self, event):
        # event.device.request_rssi()
        # print("+++")
        event.device.stream_emg(True)

    # def on_rssi(self, event):
    #   self.rssi = event.rssi
    #   self.output()
    #
    # def on_pose(self, event):
    #   self.pose = event.pose
    #   if self.pose == myo.Pose.double_tap:
    #     event.device.stream_emg(True)
    #     self.emg_enabled = True
    #   elif self.pose == myo.Pose.fingers_spread:
    #     event.device.stream_emg(False)
    #     self.emg_enabled = False
    #     self.emg = None
    #   self.output()

    def on_orientation(self, event):
        with self.lock:
            self.orientation = event.orientation
            # 这边做一个校准，可能要加一个IMU的校准按钮
            x, y, z, w = self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w
            roll = math.atan2(2 * w * x + 2 * y * z, 1 - 2 * x * x - 2 * y * y) * 180 / 3.1415926
            pitch = math.asin(max(-1, min(1, 2 * w * y - 2 * z * x))) * 180 / 3.1415926
            yaw = math.atan2(2 * w * z + 2 * x * y, 1 - 2 * y * y - 2 * z * z) * 180 / 3.1415926
            self.euler = (roll, pitch, yaw)

            self.imu_data_queue.append((event.timestamp, self.euler))
            # self.output()

    def on_emg(self, event):
        with self.lock:
            self.emg = event.emg
            # self.output()
            # print(self.emg)

    # def on_unlocked(self, event):
    #   self.locked = False
    #   self.output()

    # def on_locked(self, event):
    #   self.locked = True
    #   self.output()


class Plot(object):
    def __init__(self, listener):
        self.n = listener.n
        self.listener = listener
        self.fig = plt.figure()
        self.axes = [self.fig.add_subplot(3, 1, i) for i in range(1, 4)]
        [(ax.set_ylim([-100, 100])) for ax in self.axes]
        self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
        plt.ion()

    def update_plot(self):
        emg_data = self.listener.get_imu_data()
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


if __name__ == '__main__':
    myo.init(sdk_path="myo-sdk-win-0.9.0")
    # t0 = time.time()

    hub = myo.Hub()
    listener = Listener(100)
    with hub.run_in_background(listener.on_event):
        Plot(listener).main()

    # myo.init()
    # hub = myo.Hub()
    # listener = Listener()
    # with hub.run_in_background(listener.on_event):
    #   while True:
    #     print(listener.emg)
