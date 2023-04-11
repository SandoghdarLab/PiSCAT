import os
import unittest

from read_write_data_test import delete_directory

from piscat.InputOutput.reading_videos import video_reader
from piscat.InputOutput.write_video import write_binary, write_GIF, write_MP4

current_path = os.path.abspath(os.path.join("."))


class TestWriteBinary(unittest.TestCase):
    def setUp(self):
        self.save_path = current_path
        self.save_path = os.path.join(self.save_path, "TestData/Video")
        file_name_read = "600_100_100_f8.raw"
        file_path = os.path.join(self.save_path, file_name_read)
        self.video_data = video_reader(file_path, type="binary", img_width=100, img_height=100)

    def test_write_binary(self):
        file_name_write = "test_binary_file.raw"
        video_path = write_binary(self.save_path, file_name_write, self.video_data)
        self.assertTrue(os.path.exists(video_path), "directory is not created")
        self.assertTrue(
            os.path.exists(os.path.join(video_path, file_name_write)), "file is not saved"
        )
        delete_directory(video_path)

    def test_write_MP4(self):
        file_name_write = "test_MP4_file.mp4"
        # data = self.video_data.astype('uint8')
        video_path = write_MP4(self.save_path, file_name_write, self.video_data, jump=1)
        self.assertTrue(os.path.exists(video_path), "directory is not created")
        self.assertTrue(
            os.path.exists(os.path.join(video_path, file_name_write)), "file is not saved"
        )
        delete_directory(video_path)

    def test_write_GIF(self):
        file_name_write = "test_gif_file.gif"
        video_path = write_GIF(self.save_path, file_name_write, self.video_data, jump=1)
        self.assertTrue(os.path.exists(video_path), "directory is not created")
        self.assertTrue(
            os.path.exists(os.path.join(video_path, file_name_write)), "file is not saved"
        )
        delete_directory(video_path)


if __name__ == "__main__":
    unittest.main()
