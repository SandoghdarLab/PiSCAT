import os
import sys
import numpy as np

def fixed_length(in_list):
    """
    This function create the list with same length for all sub list.

    Parameters
    ----------
    in_list: list
        List contains some sublist.

    Returns
    -------
    tmp: list
    """
    tmp = np.empty([len(in_list), len(max(in_list, key=lambda x: len(x)))])
    for i, j in enumerate(in_list):
        tmp[i][0:len(j)] = j
    return tmp

def protein_trajectories_list2dic(v_shape_list):
    """
    From the output of ``TemporalFilter.v_trajectory``, this function converts the list to dictionary format.

    Parameters
    ----------
    v_shape_list: List of list

        | [intensity_horizontal, intensity_vertical, particle_center_intensity,
            particle_center_intensity_follow, particle_frame, particle_sigma, particle_X, particle_Y, particle_ID,
            optional(fit_intensity, fit_x, fit_y, fit_X_sigma, fit_Y_sigma, fit_Bias, fit_intensity_error,
            fit_x_error, fit_y_error, fit_X_sigma_error, fit_Y_sigma_error, fit_Bias_error)]

    Returns
    -------
    dic_all: dic
       Return dictionary similar to the following structures

       | {"#0": {'intensity_horizontal': ..., 'intensity_vertical': ..., ..., 'particle_ID': ...},
            "#1": {}, ...}

    """
    dic_all = {}

    if type(v_shape_list) is list:
        v_shape_list = np.asarray(v_shape_list)

    if type(v_shape_list) is np.ndarray:

        for n_particle in range(v_shape_list.shape[0]):
            dic_particles = {}

            if type(v_shape_list[n_particle][0]) is list:
                dic_particles['intensity_horizontal'] = np.asarray(fixed_length(v_shape_list[n_particle][0]))
                dic_particles['intensity_vertical'] = np.asarray(fixed_length(v_shape_list[n_particle][1]))
                dic_particles['center_int'] = np.asarray(v_shape_list[n_particle][2])
                dic_particles['center_int_flow'] = np.asarray(v_shape_list[n_particle][3])
                dic_particles['frame_number'] = np.asarray(v_shape_list[n_particle][4])
                dic_particles['sigma'] = np.asarray(v_shape_list[n_particle][5])
                dic_particles['x_center'] = np.asarray(v_shape_list[n_particle][6])
                dic_particles['y_center'] = np.asarray(v_shape_list[n_particle][7])
                dic_particles['particle_ID'] = np.asarray(v_shape_list[n_particle][8])
            else:
                dic_particles['intensity_horizontal'] = v_shape_list[n_particle][0].ravel()
                dic_particles['intensity_vertical'] = v_shape_list[n_particle][1].ravel()
                dic_particles['center_int'] = v_shape_list[n_particle][2].ravel()
                dic_particles['center_int_flow'] = v_shape_list[n_particle][3].ravel()
                dic_particles['frame_number'] = v_shape_list[n_particle][4].ravel()
                dic_particles['sigma'] = v_shape_list[n_particle][5].ravel()
                dic_particles['x_center'] = v_shape_list[n_particle][6].ravel()
                dic_particles['y_center'] = v_shape_list[n_particle][7].ravel()
                dic_particles['particle_ID'] = v_shape_list[n_particle][8].ravel()

            num_parameters = len(v_shape_list[n_particle])

            if num_parameters == 21:
                if type(v_shape_list[n_particle][0]) is list:
                    dic_particles['fit_intensity'] = np.asarray(v_shape_list[n_particle][9])
                    dic_particles['fit_x'] = np.asarray(v_shape_list[n_particle][10])
                    dic_particles['fit_y'] = np.asarray(v_shape_list[n_particle][11])
                    dic_particles['fit_X_sigma'] = np.asarray(v_shape_list[n_particle][12])
                    dic_particles['fit_Y_sigma'] = np.asarray(v_shape_list[n_particle][13])
                    dic_particles['fit_Bias'] = np.asarray(v_shape_list[n_particle][14])
                    dic_particles['fit_intensity_error'] = np.asarray(v_shape_list[n_particle][15])
                    dic_particles['fit_x_error'] = np.asarray(v_shape_list[n_particle][16])
                    dic_particles['fit_y_error'] = np.asarray(v_shape_list[n_particle][17])
                    dic_particles['fit_X_sigma_error'] = np.asarray(v_shape_list[n_particle][18])
                    dic_particles['fit_Y_sigma_error'] = np.asarray(v_shape_list[n_particle][19])
                    dic_particles['fit_Bias_error'] = np.asarray(v_shape_list[n_particle][20])

                else:
                    dic_particles['fit_intensity'] = v_shape_list[n_particle][9].ravel()
                    dic_particles['fit_x'] = v_shape_list[n_particle][10].ravel()
                    dic_particles['fit_y'] = v_shape_list[n_particle][11].ravel()
                    dic_particles['fit_X_sigma'] = v_shape_list[n_particle][12].ravel()
                    dic_particles['fit_Y_sigma'] = v_shape_list[n_particle][13].ravel()
                    dic_particles['fit_Bias'] = v_shape_list[n_particle][14].ravel()
                    dic_particles['fit_intensity_error'] = v_shape_list[n_particle][15].ravel()
                    dic_particles['fit_x_error'] = v_shape_list[n_particle][16].ravel()
                    dic_particles['fit_y_error'] = v_shape_list[n_particle][17].ravel()
                    dic_particles['fit_X_sigma_error'] = v_shape_list[n_particle][18].ravel()
                    dic_particles['fit_Y_sigma_error'] = v_shape_list[n_particle][19].ravel()
                    dic_particles['fit_Bias_error'] = v_shape_list[n_particle][20].ravel()
            dic_all['#'+str(n_particle)] = dic_particles
        return dic_all


