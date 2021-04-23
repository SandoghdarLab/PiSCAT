from __future__ import print_function

import numpy as np


class StatusLine():

    def __init__(self, video):
        """
        A status line from an image frame is read.
        The last line of the Photonfocus camra picture is the status line.
        All data is returned in the form of a struct, as well as the cut-frame without the status line.

        Parameters
        ----------
        video: NDArray
            Numpy array with the following form should be used for video (number of frame, width, height).
        """

        self.video = video
        self.out_video = None
        self.status_line_row_flag = False
        self.status_line_column_flag = False
        self.camera_info = {}

    def find_status_line(self):
        """
        Returns
        -------
        self.out_video: NDArray
            Video without status line.

        self.camera_info: dic
            The dictionary that illustrates the obtain information in the status line.
        """

        if np.all(self.video[0, -1, :4] == [0xFF, 0x00, 0xAA, 0x55]):
            self.status_line_column_flag = True
            self.read_status_line(self.video[0])
            print("---Status line detected in column---")
            self.camera_info['status_line_position'] = 'column'
            return self.out_video, self.camera_info

        elif np.all(self.video[0, :4, -1] == [0xFF, 0x00, 0xAA, 0x55]):
            self.status_line_row_flag = True
            self.read_status_line(self.video[0])
            print("---Status line detected in row---")
            self.camera_info['status_line_position'] = 'row'
            return self.out_video, self.camera_info

        else:
            print("---Status line does not detect---")
            self.camera_info['status_line_position'] = ''
            return self.video, self.camera_info

    def dec2hex(self, n):
        return "%X" % int(n)

    def hex2dec(self, s):
        return int(s, 16)

    def decimalToBinary(self, num, len_):
        d2b_ = np.zeros(len_)
        d2b = bin(int(num)).replace("0b", "")
        tmp_ = [int(d_) for d_ in d2b]
        diff_len = abs(len_ - len(tmp_))
        d2b_[diff_len:] = tmp_

        return d2b_

    def read_status_line(self, frame):
        """
        Parameters
        ----------
        frame: NDArray
            First frame in video.
        """

        # constants for our camera
        clock_cycle = 80e6  # Camera Clock Cycle in Hz
        nr_taps = 2  # Nr of Taps
        if self.status_line_column_flag:
            cutframe = frame[:, 0:- 1]
            self.out_video = self.video[:, 1:- 1, :]

            width = frame.shape[0]
            converter = [1, 256, 256 * 256, 256 * 256 * 256]
            StatusLine = frame[-1, :]

        elif self.status_line_row_flag:
            cutframe = frame[0:- 1, :]
            self.out_video = self.video[:, :, 1:- 1]

            width = frame.shape[1]
            converter_ = [1, 256, 256 * 256, 256 * 256 * 256]
            converter = np.asarray(converter_).astype(np.int64)
            StatusLine = frame[:, -1].astype(np.int64)

        firstbits = sum(np.multiply(StatusLine[0:4], converter))
        NrAcc = firstbits / self.hex2dec('55AA00FF')

        if (NrAcc % 1) != 0:
            print('\n... It looks like there is no Status line!?')

        self.NrAcc = NrAcc
        self.Preamble = self.dec2hex(firstbits / NrAcc)  # Preamble (0x55AA00FF)

        if width >= 8:
            ImgCnt = sum(np.multiply(StatusLine[4:8], converter)) / NrAcc  # (Mean) Image Counter
            self.camera_info['ImgCnt'] = ImgCnt

        if width >= 12:
            RealTimeCnt = sum(np.multiply(StatusLine[8:12], converter)) / NrAcc  # (Mean) Real Time Counter in Microseconds
            self.camera_info['RealTimeCnt'] = RealTimeCnt

        if width >= 16:
            MissedTrgCnt = sum(np.multiply(StatusLine[12:16], converter)) / NrAcc  # (Mean) Missed Trigger Counter
            self.camera_info['MissedTrgCnt'] = MissedTrgCnt

        if width >= 20:
            self.ImgAvgVal = sum(np.multiply(StatusLine[16:20], converter)) / NrAcc  # (Mean) Image Average Value
            self.camera_info['RealTimeCnt'] = RealTimeCnt

        if width >= 24:
            IntTimeClk = sum(np.multiply(StatusLine[20:24], converter))/ NrAcc  # Integration Time in units of clock cycles (80MHz)
            IntTime = IntTimeClk / clock_cycle * nr_taps * 1000  # Integration Time in ms
            self.camera_info['IntTimeClk'] = IntTimeClk
            self.camera_info['IntTime'] = IntTime

        if width >= 28:
            BrstTrgNr = sum(np.multiply(StatusLine[24:28], converter)) / NrAcc  # (Mean) Burst Trigger Number
            self.camera_info['BrstTrgNr'] = BrstTrgNr

        if width >= 32:
            MissedBrstTrgCnt = sum(np.multiply(StatusLine[28:32], converter)) / NrAcc  # (Mean) Missed Burst Trigger Counter
            self.camera_info['MissedBrstTrgCnt'] = MissedBrstTrgCnt

        if width >= 36:
            ROIx_start = sum(np.multiply(StatusLine[32:36], converter)) / NrAcc  # Horizontal start position of ROI (Window.X)
            self.camera_info['ROIx_start'] = ROIx_start

        if width >= 40:
            ROIx_end = sum(np.multiply(StatusLine[36:40], converter)) / NrAcc  # Horizontal end position of ROI (= Window.X + Window.W - 1)
            NrCols = ROIx_end - ROIx_start + 1
            self.camera_info['ROIx_end'] = ROIx_end
            self.camera_info['NrCols'] = NrCols

        if width >= 44:
            ROIy_start = sum(np.multiply(StatusLine[40:44], converter)) / NrAcc  # Vertical start position of ROI (Window.Y). In MROI-mode this parameter is the start position of the first ROI
            self.camera_info['ROIy_start'] = ROIy_start

        if width >= 48:
            NrRows = sum(np.multiply(StatusLine[44:48], converter)) / NrAcc + 1  # Number of rows
            ROIy_end = ROIy_start + NrRows - 1
            self.camera_info['NrRows'] = NrRows
            self.camera_info['ROIy_end'] = ROIy_end

        if width >= 52:
            TrgSrc = sum(np.multiply(StatusLine[48:52], converter)) / NrAcc # Trigger Source (0=Free Running, 1=Interface Trigger, 2=I/O Trigger)
            self.camera_info['TrgSrc'] = TrgSrc

        if width >= 56:
            DigGain = 2 ** (sum(np.multiply(StatusLine[52:56], converter)) / NrAcc)  # Digital Gain (1x, 2x, 4x, 8x)
            self.camera_info['DigGain'] = DigGain

        if width >= 60:
            DigOffset = sum(np.multiply(StatusLine[56:60], converter)) / NrAcc  # Digital Offset
            self.camera_info['DigOffset'] = DigOffset

        if width >= 64:
            CamType = sum(np.multiply(StatusLine[60:64], converter)) / NrAcc  # Camera Type Code (Our Photonfocus "MV1-D1024E-160-CL-12" is 110)
            self.camera_info['CamType'] = CamType

        if width >= 68:
            SerialNr = sum(np.multiply(StatusLine[64:68], converter)) / NrAcc  # Camera Serial Number
            self.camera_info['SerialNr'] = SerialNr

        if width >= 80:
            tmp_ = list(range(3, -13, -1))
            tmp_0 = np.float_power(2, tmp_)
            tmp_1 = np.multiply(StatusLine[76:80], converter)
            tmp_2 = int(np.sum(tmp_1)/self.NrAcc)
            tmp_3 = self.decimalToBinary(tmp_2, 16)
            tmp_4 = np.multiply(tmp_3, tmp_0)
            FineGain = np.sum(tmp_4)
            self.camera_info['FineGain'] = FineGain

        if width >= 96:
            TrgLvl = self.decimalToBinary(int(np.sum(np.multiply(StatusLine[92:96], converter))) / NrAcc, 4)  # Trigger Level: signals level of the trigger input signals. Bit 0: ExSync (CC1): Bit 1: I/O Trigger; Bit 2: CC3; Bit 3: CC4
            self.camera_info['TrgLvl'] = TrgLvl

