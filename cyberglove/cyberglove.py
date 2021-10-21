import numpy as np
import struct
import serial
import serial.tools.list_ports


def load_calibration(cal_path, n_df):
    """
    Reads a CyberGlove calibration file and returns offset and gain values.

    Gains are converted from radians to degrees.

    Parameters
    ----------
    cal_path : string
        Calibration file path.
    n_df : integer (18 or 22)
        Data glove model (18-DOF or 22-DOF).

    Returns
    -------
    offset : array, shape = (n_df,)
        Sensor offsets.
    gain : array, shape = (n_df,)
        Sensor gains.

    Notes
    -----
    The Finger1_3 and Finger5_3 values are not used as they do not correspond
    to any DOFs currently implemented in the Cyberglove. There must be also a
    bug in how DCU stores the gain parameter for Finger2_3 as this is saved in
    the Finger1_3 field. For this reason, the indexes are slightly different
    for offset and gain.

    """
    f = open(cal_path, 'r')
    lines = f.readlines()
    if n_df == 18:
        lines_idx_offset = [2, 3, 4, 5, 7, 8, 12, 13, 15, 17, 18, 20, 22, 23,
                            25, 27, 28, 29]
        lines_idx_gain = [2, 3, 4, 5, 7, 8, 12, 13, 10, 17, 18, 20, 22, 23,
                          25, 27, 28, 29]
    elif n_df == 22:
        lines_idx_offset = [2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 15, 17, 18, 19,
                            20, 22, 23, 24, 25, 27, 28, 29]
        lines_idx_gain = [2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 10, 17, 18, 19,
                          20, 22, 23, 24, 25, 27, 28, 29]
    else:
        raise ValueError("Cyberglove can be either 18-DOF or 22-DOF.")
    offset = []
    gain = []
    for line in lines_idx_offset:
        offset.append(-float(lines[line].split(' ')[6]))
    for line in lines_idx_gain:
        gain.append(float(lines[line].split(' ')[9])
                    * (180 / np.pi))  # Degrees
    offset = np.asarray(offset)
    gain = np.asarray(gain)
    return (offset, gain)


def calibrate_data(data, offset, gain):
    """
    Calibrates CyberGlove raw data.

    Parameters
    ----------
    data : array, shape = (n_df,)
        Raw CyberGlove data.
    offset : array, shape = (n_df,)
        Sensor offsets.
    gain : array, shape = (n_df,)
        Sensor offsets.

    Returns
    -------
    data : array, shape = (n_df,)
        Calibrated CyberGlove data.
    """
    return data * gain + offset


class CyberGlove(object):
    """
    Interface the Cyberglove via a serial port.

    Parameters
    ----------
    n_df : integer (18 or 22)
        Data glove model (18-DOF or 22-DOF).
    s_port : str, optional (default: None)
        Serial port name (e.g., 'COM1' in Windows). If None, the first one
        available will be used.
    baud_rate : int, optional (default: 115200)
        Baud rate.
    samples_per_read : int, optional (default: 1)
        Number of samples to read in each read operation.
    cal_path : string, optional (default: None)
        Calibration file path.

    Attributes
    ----------
    calibration_ : boolean
        True if cal_path has been provided.
    offset_ : array, shape = (n_df,)
        Sensor offsets (if cal_path is not None).
    gain_ : array, shape = (n_df,)
        Sensor gains (if cal_path is not None).
    """

    def __init__(self, n_df, s_port=None, baud_rate=115200,
                 samples_per_read=1, cal_path=None):

        # If port is not given use the first one available
        if s_port is None:
            try:
                s_port = serial.tools.list_ports.comports()[0].device
            except StopIteration:
                print("No serial ports found.")

        self.n_df = n_df
        self.s_port = s_port
        self.baud_rate = baud_rate
        self.samples_per_read = samples_per_read
        self.cal_path = cal_path

        if self.n_df == 18:
            self.__bytesPerRead = 20  # First and last bytes are reserved
        elif self.n_df == 22:
            self.__bytesPerRead = 24  # First and last bytes are reserved

        self.si = serial.Serial(port=self.s_port, baudrate=self.baud_rate,
                                timeout=1, writeTimeout=1)
        if self.cal_path is None:
            self.calibration_ = False
        else:
            self.calibration_ = True
            (self.cal_offset_, self.cal_gain_) = load_calibration(
                self.cal_path, self.n_df)

    def __del__(self):
        """Call stop() on destruct."""
        self.stop()

    def start(self):
        """Open port and flush input/output."""
        if not self.si.is_open:
            self.si.open()
            self.si.flushOutput()
            self.si.flushInput()

    def stop(self):
        """Flush input/output and close port."""
        if self.si.is_open:
            self.si.flushInput()
            self.si.flushOutput()
            self.si.close()

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available. The device does not support reading
        more than one samples at a time so this is implemented in a for loop.

        Returns
        -------
        data : ndarray, shape=(n_df, samples_per_read)
            Data read from the device. Each sensor is a row and each column
            is a point in time.
        """

        fmt = '@' + "B" * self.__bytesPerRead  # Unpacking format
        data = np.zeros((self.n_df, self.samples_per_read))
        for i in range(self.samples_per_read):
            self.si.flushInput()
            raw_data = None
            while raw_data is None:
                nb = self.si.write(bytes('\x47', 'utf'))
                if nb == 1:
                    msg = self.si.read(size=self.__bytesPerRead)
                    if len(msg) is self.__bytesPerRead:
                        raw_data = struct.unpack(fmt, msg)
                        raw_data = np.asarray(raw_data)
                        raw_data = raw_data[1:-1]  # Reserved bytes
                        if self.calibration_:
                            data[:, i] = calibrate_data(
                                raw_data,
                                self.cal_offset_,
                                self.cal_gain_)
                        else:
                            data[:, i] = raw_data

        return data
