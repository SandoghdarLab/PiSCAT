import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage
import math


class DirectionalIntensity():

    def __init__(self):
        pass

    def interpolate_pixels_along_line(self, x0, y0, x1, y1):
        """
        Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
        straight line, given two points (x0, y0) and (x1, y1)

        References
        ----------
        [1] Wikipedia article containing pseudo code that function was based off of: http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
        """
        pixels = []
        steep = abs(y1 - y0) > abs(x1 - x0)

        # Ensure that the path to be interpolated is shallow and from left to right
        if steep:
            t = x0
            x0 = y0
            y0 = t

            t = x1
            x1 = y1
            y1 = t

        if x0 > x1:
            t = x0
            x0 = x1
            x1 = t

            t = y0
            y0 = y1
            y1 = t

        dx = x1 - x0
        dy = y1 - y0
        gradient = dy / dx  # slope

        # Get the first given coordinate and add it to the return list
        x_end = round(x0)
        y_end = y0 + (gradient * (x_end - x0))
        xpxl0 = x_end
        ypxl0 = round(y_end)
        if steep:
            pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
        else:
            pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

        interpolated_y = y_end + gradient

        # Get the second given coordinate to give the main loop a range
        x_end = round(x1)
        y_end = y1 + (gradient * (x_end - x1))
        xpxl1 = x_end
        ypxl1 = round(y_end)

        # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
        for x in range(xpxl0 + 1, xpxl1):
            if steep:
                pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

            else:
                pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

            interpolated_y += gradient

        # Add the second given coordinate to the given list
        if steep:
            pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
        else:
            pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

        return pixels

    def radial_profile_app2(self, data, r):
        R = data.shape[0] // 2  # Radial profile radius
        range_arr = np.arange(-R, R + 1)
        ids = (range_arr[:, None] ** 2 + range_arr ** 2).ravel()
        count = np.bincount(ids)

        R0 = R + 1
        dists = np.unique(r[:R0, :R0][np.tril(np.ones((R0, R0), dtype=bool))])

        mean_data = (np.bincount(ids, data.ravel()) / count)[count != 0]
        return dists, mean_data

    def plot_directional_intensity(self, data, origin=None):
        """
        Makes a cicular histogram showing average intensity binned by direction
        from "origin" for each band in "data" (a 3D numpy array). "origin" defaults
        to the center of the image.
        """

        def intensity_rose(theta, band, color):
            theta, band = theta.flatten(), band.flatten()
            intensities, theta_bins = self.bin_by(band, theta)
            mean_intensity = map(np.mean, intensities)
            width = np.diff(theta_bins)[0]
            plt.bar(theta_bins, mean_intensity, width=width, color=color)
            plt.xlabel(color + ' Band')
            plt.yticks([])

        # Make cartesian coordinates for the pixel indicies
        # (The origin defaults to the center of the image)
        x, y = self.index_coords(data, origin)

        # Convert the pixel indices into polar coords.
        r, theta = self.cart2polar(x, y)

        # Unpack bands of the image
        # red, green, blue = data.T
        red = data.T
        green = data.T
        blue = data.T
        # Plot...
        plt.figure()

        plt.subplot(2, 2, 1, projection='polar')
        intensity_rose(theta, red, 'Red')

        plt.subplot(2, 2, 2, projection='polar')
        intensity_rose(theta, green, 'Green')

        plt.subplot(2, 1, 2, projection='polar')
        intensity_rose(theta, blue, 'Blue')

        plt.suptitle('Average intensity as a function of direction')

    def plot_polar_image(self, data, origin=None):
        """Plots an image reprojected into polar coordinages with the origin
        at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
        polar_grid, r, theta = self.reproject_image_into_polar(data, origin)
        plt.figure()
        plt.imshow(polar_grid, extent=(theta.min(), theta.max(), r.max(), r.min()))
        plt.axis('auto')
        plt.ylim(plt.ylim()[::-1])
        plt.xlabel('Theta Coordinate (radians)')
        plt.ylabel('R Coordinate (pixels)')
        plt.title('Image in Polar Coordinates')

    def index_coords(self, data, origin=None):
        """Creates x & y coords for the indicies in a numpy array "data".
        "origin" defaults to the center of the image. Specify origin=(0,0)
        to set the origin to the lower left corner of the image."""
        ny, nx = data.shape[:2]
        if origin is None:
            origin_x, origin_y = nx // 2, ny // 2
        else:
            origin_x, origin_y = origin
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x -= origin_x
        y -= origin_y
        return x, y

    def cart2polar(self, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return r, theta

    def polar2cart(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def bin_by(self, x, y, nbins=30):
        """Bin x by y, given paired observations of x & y.
        Returns the binned "x" values and the left edges of the bins."""
        bins = np.linspace(y.min(), y.max(), nbins + 1)
        # To avoid extra bin for the max value
        bins[-1] += 1

        indicies = np.digitize(y, bins)

        output = []
        for i in range(1, len(bins)):
            output.append(x[indicies == i])

        # Just return the left edges of the bins
        bins = bins[:-1]

        return output, bins

    def reproject_image_into_polar(self, data, origin=None):
        """Reprojects a 3D numpy array ("data") into a polar coordinate system.
        "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
        ny, nx = data.shape[:2]
        if origin is None:
            origin = (nx // 2, ny // 2)

        # Determine that the min and max r and theta coords will be...
        x, y = self.index_coords(data, origin=origin)
        r, theta = self.cart2polar(x, y)

        # Make a regular (in polar space) grid based on the min and max r & theta
        r_i = np.linspace(r.min(), r.max(), nx)
        theta_i = np.linspace(theta.min(), theta.max(), ny)
        theta_grid, r_grid = np.meshgrid(theta_i, r_i)

        # Project the r and theta grid back into pixel coordinates
        xi, yi = self.polar2cart(r_grid, theta_grid)
        xi += origin[0]  # We need to shift the origin back to
        yi += origin[1]  # back to the lower-left corner...
        xi, yi = xi.flatten(), yi.flatten()
        coords = np.vstack((xi, yi))  # (map_coordinates requires a 2xn array)

        # Reproject each band individually and the restack
        # (uses less Memory than reprojection the 3-dimensional array in one step)
        bands = []
        for band in data.T:
            zi = sp.ndimage.map_coordinates(band, coords, order=1)
            bands.append(zi.reshape((nx, ny)))
        output = np.dstack(bands)
        return output, r_i, theta_i