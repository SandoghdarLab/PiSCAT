from piscat.Trajectory.particle_linking import Linking
from piscat.Localization import localization_filtering
from piscat.Trajectory import TemporalFilter

import os
import pickle


def load_fixture(filename):
    """Loads a fixture from file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


current_path = os.path.abspath(os.path.join('..'))


if __name__ == "__main__":

    directory_path = os.path.join(current_path, 'TestData/Video/')
    file_name_save = os.path.join(directory_path, 'test_fit_Gaussian2D_wrapper.pck')
    psf_dataframe = load_fixture(file_name_save)
    linking_ = Linking()
    linked_psf = linking_.create_link(psf_position=psf_dataframe, search_range=2, memory=10)
    file_name_save = os.path.join(directory_path, 'test_create_link.pck')

    with open(file_name_save, 'wb') as file:
        pickle.dump(linked_psf, file)

    sorted_linking_df = linking_.sorting_linking(linked_psf)
    file_name_save = os.path.join(directory_path, 'test_sort_linking.pck')
    with open(file_name_save, 'wb') as file:
        pickle.dump(linked_psf, file)

    spatial_filters = localization_filtering.SpatialFilter()
    psf_filtered = spatial_filters.outlier_frames(linked_psf, threshold=20)
    psf_filtered = spatial_filters.dense_PSFs(psf_filtered, threshold=0)
    psf_filtered = spatial_filters.symmetric_PSFs(psf_filtered, threshold=0.7)
    file_name_save = os.path.join(directory_path, 'test_localization_input_video.pck')
    video = load_fixture(file_name_save)
    batch_size = 3
    test_obj = TemporalFilter(video=video, batchSize=batch_size)

    all_trajectories, linked_PSFs_filter, his_all_particles = test_obj.v_trajectory(df_PSFs=psf_filtered,
                                                                                    threshold_min=2,
                                                                                    threshold_max=2 * batch_size)
    file_name_save = os.path.join(directory_path, 'test_v_trajectory_all_trajectories.pck')
    with open(file_name_save, 'wb') as file:
        pickle.dump(all_trajectories, file)

    file_name_save = os.path.join(directory_path, 'test_v_trajectory_linked_PSFs_filter.pck')
    with open(file_name_save, 'wb') as file:
        pickle.dump(linked_PSFs_filter, file)

    file_name_save = os.path.join(directory_path, 'his_all_particles.pck')
    with open(file_name_save, 'wb') as file:
        pickle.dump(his_all_particles, file)
