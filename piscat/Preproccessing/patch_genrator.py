from __future__ import print_function
import numpy as np
from tqdm.autonotebook import tqdm


class ImagePatching():

    def __init__(self, depth=0, width=0, height=0, depth_overlap=0, width_overlap=0, height_overlap=0):

        self.total_start_position = None
        self.weight_matrix = None
        self.patches = None

        self.size = [depth, width, height, depth_overlap, width_overlap, height_overlap]
        self.depth_step = depth - depth_overlap
        self.width_step = width - width_overlap
        self.height_step = height - height_overlap

    def split_video(self, video):
        return [video[f:f+self.depth_step, w:w+self.width_step, h:h+self.height_step]
                for f in range(0, video.shape[0], self.depth_step)
                for w in range(0, video.shape[1], self.width_step)
                for h in range(0, video.shape[2], self.height_step)]

    def inverse_split(self, split_video, video_size):
        number_d_patch = int(video_size[0]/self.size[0])
        number_w_patch = int(video_size[1]/self.size[1])
        number_h_patch = int(video_size[2]/self.size[2])
        blocks = None
        index = 0
        for d_ in range(number_d_patch):
            temp1 = None

            for h_ in range(number_h_patch):
                temp0 = None

                for w_ in range(number_w_patch):

                    if temp0 is None:
                        temp0 = split_video[index]
                    else:
                        temp0 = np.concatenate((temp0, split_video[index]), axis=2)

                    if index <= len(split_video):
                        index = index + 1
                    else:
                        print('finsish!!!')
                        break
                if temp1 is None:
                    temp1 = temp0
                else:
                    temp1 = np.concatenate((temp1, temp0), axis=1)

            if blocks is None:
                blocks = temp1
            else:
                blocks = np.concatenate((blocks, temp1), axis=0)
        return blocks

    def split_weight_matrix(self, video, patch_size=16, strides=4):
        self.total_start_position = []
        self.weight_matrix = np.zeros((video.shape[1], video.shape[2]), dtype=np.int)
        self.patches = []
        for i_ in range(0, video.shape[1] - patch_size + strides, strides):
            for j_ in range(0, video.shape[2] - patch_size + strides, strides):
                sRow = i_
                eRow = i_ + patch_size

                sColumn = j_
                eColumn = j_ + patch_size

                list_start_position = [sRow, eRow, sColumn, eColumn]
                self.total_start_position.append(list_start_position)
                self.weight_matrix[sRow:eRow, sColumn:eColumn] = self.weight_matrix[sRow:eRow, sColumn:eColumn] + 1
                patch = video[:, sRow:eRow, sColumn:eColumn]
                self.patches.append(patch)
        return self.patches

    def reconstruction_weight_matrix(self, video_shape, new_patch):
        self.reconstruction_video = np.zeros(video_shape, dtype=np.float64)
        for i_ in tqdm(range(len(self.total_start_position))):
            sRow = self.total_start_position[i_][0]
            eRow = self.total_start_position[i_][1]

            sColumn = self.total_start_position[i_][2]
            eColumn = self.total_start_position[i_][3]

            self.reconstruction_video[:, sRow:eRow, sColumn:eColumn] = self.reconstruction_video[:, sRow:eRow, sColumn:eColumn] + new_patch[i_]

        self.weight_matrix_3D = np.broadcast_to(self.weight_matrix, (self.reconstruction_video.shape[0], video_shape[1], video_shape[2]))
        self.reconstruction_video = np.divide(self.reconstruction_video, self.weight_matrix_3D)
        return self.reconstruction_video

    def reconstruction_weight_matrix_kernel(self, i_):
        sRow = self.total_start_position[i_][0]
        eRow = self.total_start_position[i_][1]

        sColumn = self.total_start_position[i_][2]
        eColumn = self.total_start_position[i_][3]

        self.reconstruction_video[:, sRow:eRow, sColumn:eColumn] = self.reconstruction_video[:, sRow:eRow, sColumn:eColumn] + self.patches[i_]




