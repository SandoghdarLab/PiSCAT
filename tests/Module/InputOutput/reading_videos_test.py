from piscat.InputOutput.reading_videos import video_reader, read_tif_iterate, DirectoryType
import os
import unittest


current_path = os.path.abspath(os.path.join('.'))


class TestVideoReader(unittest.TestCase):
    def setUp(self):
        # self.save_path = os.path.dirname(current_path)
        self.save_path = current_path
        self.save_path = os.path.join(self.save_path, 'Data')

    def test_read_tif(self):
        file_name = '600_100_100_f8.tif'
        file_path = self.save_path = os.path.join(self.save_path, file_name)
        video = video_reader(file_path, type='tif')
        self.assertTrue(video.dtype == 'float32')

    def test_read_tif_iterate(self):
        file_name = '600_100_100_f8.tif'
        file_path = self.save_path = os.path.join(self.save_path, file_name)
        video = read_tif_iterate(file_path)
        self.assertTrue(video.dtype == 'float32')

    def test_read_avi(self):
        file_name = '600_100_100_f8.avi'
        file_path = self.save_path = os.path.join(self.save_path, file_name)
        video = video_reader(file_path, type='avi')
        self.assertTrue(video.dtype == 'uint8')

    def test_read_binary(self):
        file_name = '600_100_100_f8.raw'
        file_path = os.path.join(self.save_path, file_name)
        video = video_reader(file_path, type='binary', img_width=100, img_height=100)
        self.assertTrue(video.dtype == 'float64')


class TestDirectoryType(unittest.TestCase):
    def setUp(self):
        # self.save_path = os.path.dirname(current_path)
        self.save_path = current_path
        self.save_path = os.path.join(self.save_path, 'Data')

    def test_return_df(self):
        test_obj = DirectoryType(self.save_path, '.tif')
        df = test_obj.return_df()
        self.assertTrue(len(df) == 1)

    def test_get_list_of_files(self):
        test_obj = DirectoryType(self.save_path, '.tif')
        path_list = test_obj.get_list_of_files(self.save_path)
        self.assertTrue(len(path_list) > 1)


if __name__ == '__main__':
    unittest.main()
