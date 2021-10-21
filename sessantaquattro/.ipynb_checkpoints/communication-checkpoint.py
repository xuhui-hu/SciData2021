#!python3
# -------------------------------------------------------
# Module to connect to Sessantaquattro Bio signal data logger
#
# import datetime
# import multiprocessing
# import socket  # we will need this for establishing the communication with Sessantaquattro
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation

CONVERSION_FACTOR = 0.000286  # conversion factor needed to get values in mV


# Convert integer to bytes
def integer_to_bytes(command):
    return int(command).to_bytes(2, byteorder="big")


# Convert byte-array value to an integer value and apply two's complement
def convert_bytes_to_int(i, bytes_value, bytes_in_sample):
    # value = None
    if bytes_in_sample == 2:
        # Combine 2 bytes to a 16 bit integer value
        try:
            if len(bytes_value) == 0:
                value = 0
                print("channel %s = 0" % i)
            else:
                value = \
                    bytes_value[0] * 256 + \
                    bytes_value[1]
        except IndexError as e:
            print(e, len(bytes_value), bytes_value)

        # See if the value is negative and make the two's complement
        if value >= 32768:
            value -= 65536
    elif bytes_in_sample == 3:
        # Combine 3 bytes to a 24 bit integer value
        value = \
            bytes_value[0] * 65536 + \
            bytes_value[1] * 256 + \
            bytes_value[2]
        # See if the value is negative and make the two's complement
        if value >= 8388608:
            value -= 16777216
    else:
        raise Exception(
            "Unknown bytes_in_sample value. Got: {}, "
            "but expecting 2 or 3".format(bytes_in_sample))
    return value


# Create the binary command which is sent to Sessantaquattro
# to start or stop the communication with wanted data logging setup
def create_bin_command(fsamp=2, nch=3, mode=0,
                       hres=0, hpf=1, ext=0,
                       trig=0):
    command = 0
    # command = command + go  # go is set in function "connect_to_sq" or "disconnect_to_sq"
    # command = command + rec * 2  # rec = 0
    command = command + trig * 4
    command = command + ext * 16
    command = command + hpf * 64
    command = command + hres * 128
    command = command + mode * 256
    command = command + nch * 2048
    command = command + fsamp * 8192
    # command = command + getset * 32768  # getset = 0

    # number_of_channels = None
    # sample_frequency = None
    # bytes_in_sample = None

    if nch == 0:
        if mode == 1:
            number_of_channels = 8
        else:
            number_of_channels = 12
    elif nch == 1:
        if mode == 1:
            number_of_channels = 12
        else:
            number_of_channels = 20
    elif nch == 2:
        if mode == 1:
            number_of_channels = 20
        else:
            number_of_channels = 36
    elif nch == 3:
        if mode == 1:
            number_of_channels = 36
        else:
            number_of_channels = 68
    else:
        raise Exception('Wrong value for nch. Got: {0}', nch)

    if fsamp == 0:
        if mode == 3:
            sample_frequency = 2048
        else:
            sample_frequency = 500
    elif fsamp == 1:
        if mode == 3:
            sample_frequency = 4000
        else:
            sample_frequency = 1000
    elif fsamp == 2:
        if mode == 3:
            sample_frequency = 8000
        else:
            sample_frequency = 2048
    elif fsamp == 3:
        if mode == 3:
            sample_frequency = 16000
        else:
            sample_frequency = 4000
    else:
        raise Exception('wrong value for fsamp. Got: {fsamp}', fsamp)

    if hres == 1:
        bytes_in_sample = 3  # 24 bits
    else:
        bytes_in_sample = 2

    if (
            not number_of_channels or
            not sample_frequency or
            not bytes_in_sample):
        raise Exception(
            "Could not set number_of_channels "
            "and/or and/or bytes_in_sample")

    return (command,
            number_of_channels,
            sample_frequency,
            bytes_in_sample)


# Convert channels from bytes to integers
def bytes_to_integers(
        sample_from_channels_as_bytes,
        number_of_channels,
        bytes_in_sample, batch,
        output_milli_volts):
    batchSize = number_of_channels * bytes_in_sample
    data = []
    for i in range(batch):
        channel_values = []
        # Separate channels from byte-string. One channel has
        # "bytes_in_sample" many bytes in it.
        for channel_index in range(number_of_channels):
            channel_start = channel_index * bytes_in_sample
            channel_end = (channel_index + 1) * bytes_in_sample
            channel = sample_from_channels_as_bytes[batchSize * i + channel_start:batchSize * i + channel_end]

            # Convert channel's byte value to integer
            value = convert_bytes_to_int(channel_index, channel, bytes_in_sample)

            # Convert bio measurement channels to milli volts if needed
            # The 4 last channels (Auxiliary and Accessory-channels)
            # are not to be converted to milli volts
            if output_milli_volts and channel_index < (number_of_channels - 4):
                value *= CONVERSION_FACTOR
            channel_values.append(value)
        data.append(channel_values)
    return data  # np.array(data)


#     Read raw byte stream from data logger. Read one sample from each
#     channel. Each channel has 'bytes_in_sample' many bytes in it.
def read_raw_bytes(connection, number_of_all_channels, bytes_in_sample, batch):
    buffer_size = number_of_all_channels * bytes_in_sample * batch
    # buffer_size = 1024
    new_bytes = connection.recv(buffer_size)
    return new_bytes


def read_data(connection, number_of_channels, bytes_in_sample,
              batch=1, output_milli_volts=True):
    sample_from_channels_as_bytes = read_raw_bytes(connection, number_of_channels,
                                                   bytes_in_sample, batch)
    dataTemp = bytes_to_integers(sample_from_channels_as_bytes,
                                 number_of_channels,
                                 bytes_in_sample, batch,
                                 output_milli_volts)
    return dataTemp


# Connect to Sessantaquattro's TCP socket and send start command
def connect_to_sq(
        sq_socket,
        ip_address,
        port,
        command, go=1):
    sq_socket.bind((ip_address, port))
    sq_socket.listen(1)
    print('waiting for connection...')
    conn, addr = sq_socket.accept()
    print('Succeed into connection from address: {0}'.format(addr))
    # print("{0:b}".format(command + 1).encode())
    conn.send(integer_to_bytes(command + go))  # go = 1
    return conn


# Disconnect from Sessantaquattro by sending a stop command
def disconnect_from_sq(conn, command):
    if conn is not None:
        conn.send(integer_to_bytes(command))  # go = 0
        conn.shutdown(2)
        conn.close()
        print("Stop the connection.")
    else:
        raise Exception(
            "Can't disconnect because the"
            "connection is not established")
