import numpy as np
from utils import rotate, is_inside_triangle
import random
from scipy.stats import qmc
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib
import yaml
from box import Box
from scipy.spatial.distance import cdist, pdist


class LayoutGenerator:
    """ Layout generation class, generates wind wind farm layouts.
        Needs to be a class to obtain a consistent sobol sequence across all the parameters.
        
        kwargs:
        min_turbines: int, the minimum number of turbines in the farm
        max_turbines: int, the maximum number of turbines in the farm
        min_rotor_dist: float, the minimum distance between turbines
        max_rotor_dist: float, the maximum distance between turbines
        min_farm_lw_ratio: float, the minimum length-width ratio of the farm
        max_farm_lw_ratio: float, the maximum length-width ratio of the farm
        rotor_diameter: float, the diameter of the rotor        
    """
    def __init__(self, **kwargs):
        # check that kwargs contain the necessary parameters
        assert {'min_turbines', 'max_turbines', 'min_rotor_dist', 'max_rotor_dist', 'min_farm_lw_ratio', 'max_farm_lw_ratio', 'rotor_diameter', 'max_yaw', 'min_yaw', 'probability_operating_on', 'probability_operating_off'}.issubset(kwargs)
        
        # assign the parameters
        self.min_turbines = kwargs['min_turbines']
        self.max_turbines = kwargs['max_turbines']
        self.min_rotor_dist = kwargs['min_rotor_dist']
        self.max_rotor_dist = kwargs['max_rotor_dist']
        self.min_farm_lw_ratio = kwargs['min_farm_lw_ratio']
        self.max_farm_lw_ratio = kwargs['max_farm_lw_ratio']
        self.rotor_diameter = kwargs['rotor_diameter']
        self.max_yaw_angle = kwargs['max_yaw']
        self.min_yaw_angle = kwargs['min_yaw']
        self.probability_operating_on = kwargs['probability_operating_on']
        self.probability_operating_off = kwargs['probability_operating_off']
        
        # initialize Sobol sampler for consistent sampling across independent parameters
        self.sampler = qmc.Sobol(d=3, scramble=True)
    
    def generate_layouts(self, num_layouts: int):
        """ Returns a list with the generated layouts.""" 
        # draw sobol samples
        samples = self.sampler.random_base2(m=int(np.ceil(np.log2(num_layouts))))[:num_layouts]
        
        # sample the number of turbines and the rotor minimum distance
        n_points = samples[:, 0] * (self.max_turbines - self.min_turbines) + self.min_turbines
        n_points = n_points.astype(int)
        rotor_dists = samples[:, 1] * (self.max_rotor_dist - self.min_rotor_dist) + self.min_rotor_dist
        min_dists = self.rotor_diameter * rotor_dists
        
        # sample the width-length ratio of the farm
        farm_lw_ratios = samples[:, 2] * (self.max_farm_lw_ratio - self.min_farm_lw_ratio) + self.min_farm_lw_ratio
        
        # generate the layouts
        layouts = []
        for i in range(num_layouts):
            layout = self.generate_random_layoutv2(n_points[i], farm_lw_ratios[i], min_dists[i])
            coords = layout['base_coords']
            
            # pick at random the type of layout
            form = random.choice(['ellipse', 'triangle', 'circles', 'rectangle'])
            if form == 'ellipse':
                coords = coords[layout['elliptical_mask']]
            elif form == 'triangle':
                coords = coords[layout['triangle_mask']]
            elif form == 'circles':
                coords = coords[layout['circles_mask']]
            else:
                coords = coords

            yaw_angles = self.generate_random_yaw_angles(len(coords))
            operating_modes = self.generate_random_operating_modes(len(coords))
            
            layouts.append({'coords': coords, 'form': form, 'min_dist': min_dists[i], 'yaw_angles': yaw_angles, 'operating_modes': operating_modes})
            
        return layouts
    
    def generate_random_layout(self, n_points, farm_lw_ratio, min_dist):
        """ Generate one random wind farm layout given the number of points, the length, the width of the 
            farm and the minimum distance between turbines. Returns a dictionary with the layout and the masks
            for the different shapes.
        """
        # creating the initial rectangular domain based on farm aspect ratio
        length = np.ceil(n_points * min_dist)
        width = farm_lw_ratio * length
        num_y = np.int32(np.sqrt(n_points / farm_lw_ratio)) + 1
        num_x = np.int32(n_points / num_y) + 1

        # create regularly spaced points for the boundary of the overall rectangular domain
        x = np.linspace(0., length, num_x, dtype=np.float32)
        y = np.linspace(0., width, num_y, dtype=np.float32)

        # compute the resulting spacing between points
        init_dist = np.min((x[1] - x[0], y[1] - y[0]))
        
        # compute the max allowable movement for the points such that the minimum distance is respected
        max_movement = (init_dist - min_dist) / 2

        # create a smaller rectangular domain reducing spacing between points
        x1 = np.linspace(0. + max_movement, length - max_movement, num_x, dtype=np.float32)
        y1 = np.linspace(0. + max_movement, width - max_movement, num_y, dtype=np.float32)
        
        # the base coordinates of the turbines are the meshgrid of the smaller rectangular domain
        base_coords = np.stack(np.meshgrid(x1, y1), -1).reshape(-1, 2)
        
        # delete random points from the base coordinates if length exceeds n_points
        if len(base_coords) > n_points:
            # pick random indices to delete
            indices = np.random.choice(len(base_coords), len(base_coords) - n_points, replace=False)
            base_coords = np.delete(base_coords, indices, axis=0)
            
        # recompute the resulting spacing between points
        init_dist1 = np.min((x1[1] - x1[0], y1[1] - y1[0]))
        
        # recompute the max allowable movement for the points such that the minimum distance is respected
        max_movement1 = (init_dist1 - min_dist) / 2

        # perturb points by the max allowable movement
        noise = np.random.uniform(low=-max_movement1, high=max_movement1, size=(len(base_coords), 2))
        base_coords += noise
        
        # randomly rotate the rectangle around (0,0)
        alpha = random.uniform(-np.pi/4, np.pi/4)
        base_coords = rotate((0, 0), base_coords, alpha)

        # creating the elliptical mask
        theta = np.arange(0, 2 * np.pi, 0.01)
        v = random.choice([2, 4, 6])
        a = length / v
        b = width / 2
        x0, y0 = rotate((0, 0), (length / 2, width / 2), alpha)
        extra = random.choice([np.pi / 8, np.pi / 16, 0, - np.pi / 16, -np.pi / 8]) # extra random rotation
        s1 = ((a**2) * (np.sin(alpha - extra)**2) + (b**2) * (np.cos(alpha - extra)**2)) * (base_coords[:, 0] - x0)**2
        s2 = 2 * (b**2 - a ** 2) * np.sin(alpha - extra) * np.cos(alpha - extra) * (base_coords[:, 0] - x0) * ( base_coords[:, 1] - y0)
        s3 = ((a**2) * (np.cos(alpha - extra)**2) + (b**2) * (np.sin(alpha - extra)**2)) * (base_coords[:, 1] - y0)**2
        elliptical_mask = s1 + s2 + s3 < (a**2) * (b**2)
        
        # create the ellipse boundary
        xpos = (a) * np.cos(theta)
        ypos = (b) * np.sin(theta)
        new_xpos = x0 + (xpos) * np.cos(-alpha + extra) + (ypos) * np.sin(-alpha + extra)
        new_ypos = y0 + (-xpos) * np.sin(-alpha + extra) + (ypos) * np.cos(-alpha + extra)
        ellipse_boundary = np.array([new_xpos, new_ypos])

        # creating the triangular mask
        x11 = length / 2
        y11 = width
        x21 = 0
        y21 = 0
        x31 = length
        y31 = 0
        x = base_coords[:, 0]
        y = base_coords[:, 1]
        extra = random.choice([np.pi / 8, np.pi / 16, 0, - np.pi / 16, -np.pi / 8]) # extra random rotation
        x1, y1 = rotate((0, 0), (x11, y11), alpha + extra)
        x2, y2 = rotate((0, 0), (x21, y21), alpha + extra)
        x3, y3 = rotate((0, 0), (x31, y31), alpha + extra)
        triangle_mask = [is_inside_triangle(x1, y1, x2, y2, x3, y3, xp, yp) for xp, yp in zip(x, y)]
        triangle_boundary = np.array([[x1, y1], [x2, y2], [x3, y3]])

        # creating the small circles masks
        x = base_coords[:, 0]
        y = base_coords[:, 1]

        random_turb = random.choice(base_coords) # random turbine choice
        random_turb2 = random.choice(base_coords) # random turbine choice
        random_turb3 = random.choice(base_coords) # random turbine choice
        radius = length / 4

        circle_mask1 = (x - random_turb[0]) ** 2 + (y - random_turb[1]) ** 2 < radius ** 2
        circle_mask2 = (x - random_turb2[0]) ** 2 + (y - random_turb2[1]) ** 2 < radius ** 2
        circle_mask3 = (x - random_turb3[0]) ** 2 + (y - random_turb3[1]) ** 2 < radius ** 2
        
        circles_mask = circle_mask1 + circle_mask2 + circle_mask3
        circles_centers = [random_turb, random_turb2, random_turb3]
        
        # return the layout as a dictionary
        output_dict = {'base_coords': base_coords, 'ellipse': ellipse_boundary, 'elliptical_mask': elliptical_mask,
                       'triangle': triangle_boundary, 'triangle_mask': triangle_mask, 'circles_mask': circles_mask,
                       'circles_centers': circles_centers, 'circles_radius': radius, 'width': width, 'length': length, 
                       'alpha': alpha}

        return output_dict
    
    def generate_random_layoutv2(self, n_points, farm_lw_ratio, min_dist):
        """ Generate one random wind farm layout given the number of points, the length, the width of the 
            farm and the minimum distance between turbines. Returns a dictionary with the layout and the masks
            for the different shapes.
        """
        # creating the initial rectangular domain based on farm aspect ratio
        num_y = np.int32(np.sqrt(n_points / farm_lw_ratio))
        num_x = np.int32(n_points / num_y)
        width = np.ceil((num_y - 1) * 2.5 * min_dist)
        length = np.ceil((num_x - 1) * 2.5 * min_dist)

        # create regularly spaced points for the boundary of the overall rectangular domain
        x = np.linspace(0., length, num_x, dtype=np.float32)
        y = np.linspace(0., width, num_y, dtype=np.float32)

        base_coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)
        
        # delete random points from the base coordinates if length exceeds n_points
        if len(base_coords) > n_points:
            # pick random indices to delete
            indices = np.random.choice(len(base_coords), len(base_coords) - n_points, replace=False)
            base_coords = np.delete(base_coords, indices, axis=0)

        # Perturb all points with a random noise
        r = min_dist * np.sqrt(np.random.uniform(0, 1, (len(base_coords), 1)))
        theta = np.random.uniform(0, 2 * np.pi, (len(base_coords), 1))
        perturbations = np.concatenate((r * np.cos(theta), r * np.sin(theta)), axis=1)
        base_coords += perturbations

        # compute the resulting spacing between points
        min_proximity = np.min(pdist(base_coords))

        # compute scaling factor
        factor = min_dist / min_proximity

        # scale the coordinates
        base_coords *= factor
        width *= factor
        length *= factor

        # all_closest_distances = np.zeros(len(base_coords))

        # # For each point, find the distance to the nearest other point
        # for i in range(len(base_coords)):
        #     # Get the distances from the current point to all other points
        #     distances = cdist(base_coords[i].reshape(1, -1), base_coords)
        #     # Remove the distance to the current point
        #     distances = np.delete(distances, i)
        #     # Find the minimum distance
        #     min_distance = np.min(distances)
        #     all_closest_distances[i] = min_distance

        # print('min, avg and max closest distances', min(all_closest_distances), np.mean(all_closest_distances), max(all_closest_distances))

        # min_min = np.min(all_closest_distances)
        # max_min = np.max(all_closest_distances)
        
        # randomly rotate the rectangle around (0,0)
        alpha = random.uniform(-np.pi/4, np.pi/4)
        base_coords = rotate((0, 0), base_coords, alpha)

        # creating the elliptical mask
        theta = np.arange(0, 2 * np.pi, 0.01)
        v = random.choice([2, 4, 6])
        a = length / v
        b = width / 2
        x0, y0 = rotate((0, 0), (length / 2, width / 2), alpha)
        extra = random.choice([np.pi / 8, np.pi / 16, 0, - np.pi / 16, -np.pi / 8]) # extra random rotation
        s1 = ((a**2) * (np.sin(alpha - extra)**2) + (b**2) * (np.cos(alpha - extra)**2)) * (base_coords[:, 0] - x0)**2
        s2 = 2 * (b**2 - a ** 2) * np.sin(alpha - extra) * np.cos(alpha - extra) * (base_coords[:, 0] - x0) * ( base_coords[:, 1] - y0)
        s3 = ((a**2) * (np.cos(alpha - extra)**2) + (b**2) * (np.sin(alpha - extra)**2)) * (base_coords[:, 1] - y0)**2
        elliptical_mask = s1 + s2 + s3 < (a**2) * (b**2)
        
        # create the ellipse boundary
        xpos = (a) * np.cos(theta)
        ypos = (b) * np.sin(theta)
        new_xpos = x0 + (xpos) * np.cos(-alpha + extra) + (ypos) * np.sin(-alpha + extra)
        new_ypos = y0 + (-xpos) * np.sin(-alpha + extra) + (ypos) * np.cos(-alpha + extra)
        ellipse_boundary = np.array([new_xpos, new_ypos])

        # creating the triangular mask
        x11 = length / 2
        y11 = width
        x21 = 0
        y21 = 0
        x31 = length
        y31 = 0
        x = base_coords[:, 0]
        y = base_coords[:, 1]
        extra = random.choice([np.pi / 8, np.pi / 16, 0, - np.pi / 16, -np.pi / 8]) # extra random rotation
        x1, y1 = rotate((0, 0), (x11, y11), alpha + extra)
        x2, y2 = rotate((0, 0), (x21, y21), alpha + extra)
        x3, y3 = rotate((0, 0), (x31, y31), alpha + extra)
        triangle_mask = [is_inside_triangle(x1, y1, x2, y2, x3, y3, xp, yp) for xp, yp in zip(x, y)]
        triangle_boundary = np.array([[x1, y1], [x2, y2], [x3, y3]])

        # creating the small circles masks
        x = base_coords[:, 0]
        y = base_coords[:, 1]

        random_turb = random.choice(base_coords) # random turbine choice
        random_turb2 = random.choice(base_coords) # random turbine choice
        random_turb3 = random.choice(base_coords) # random turbine choice
        radius = length / 4

        circle_mask1 = (x - random_turb[0]) ** 2 + (y - random_turb[1]) ** 2 < radius ** 2
        circle_mask2 = (x - random_turb2[0]) ** 2 + (y - random_turb2[1]) ** 2 < radius ** 2
        circle_mask3 = (x - random_turb3[0]) ** 2 + (y - random_turb3[1]) ** 2 < radius ** 2
        
        circles_mask = circle_mask1 + circle_mask2 + circle_mask3
        circles_centers = [random_turb, random_turb2, random_turb3]
        
        # return the layout as a dictionary
        output_dict = {'base_coords': base_coords, 'ellipse': ellipse_boundary, 'elliptical_mask': elliptical_mask,
                        'triangle': triangle_boundary, 'triangle_mask': triangle_mask, 'circles_mask': circles_mask,
                        'circles_centers': circles_centers, 'circles_radius': radius, 'width': width, 'length': length, 
                        'alpha': alpha}

        return output_dict
    
    def generate_random_yaw_angles(self, n_points):
        """ Generate random yaw angles for the turbines in the farm. """
        return np.random.uniform(self.min_yaw_angle, self.max_yaw_angle, n_points)
    
    def generate_random_operating_modes(self, n_points):
        """ Generate random operating modes for the turbines in the farm. """
        return np.random.choice([0, 1], n_points, p=[self.probability_operating_off, self.probability_operating_on])
        
    def plot(self, layout):
        """ Plotting function to visualize a wind farm layout with all the possible shapes. """
        
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['axes.unicode_minus'] = False

        coords = layout['base_coords']
        length = layout['length']
        width = layout['width']
        alpha = layout['alpha']
        ellipse = layout['ellipse']
        ellipse_mask = layout['elliptical_mask']
        triangle = layout['triangle']
        triangle_mask = layout['triangle_mask']
        cicles_radius = layout['circles_radius']
        circle_centers = layout['circles_centers']
        circles_mask = layout['circles_mask']
        
            
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.scatter(coords[:, 0], coords[:, 1], s=150, c='black', marker="2",
                    linewidth=1, label=str(len(coords[:, 0])) + ' WT')
        r = Rectangle((0, 0), length, width, linestyle='--', fill=False)
        t2 = matplotlib.transforms.Affine2D().rotate(alpha) + ax.transData
        r.set_transform(t2)
        plt.scatter(coords[ellipse_mask][:, 0], coords[ellipse_mask][:, 1], s=150, c='red', marker="2",
                    linewidth=1, label='Inside ellipse')
        plt.plot(ellipse[0, :], ellipse[1, :], 'r', linestyle='--')
        ax.add_patch(r)
        plt.scatter(coords[triangle_mask][:, 0], coords[triangle_mask][:, 1], s=150, c='blue', marker="2",
                    linewidth=1, label='Inside triangle')
        t1 = plt.Polygon(triangle[:3, :], color='Blue', linestyle='--', fill=False)
        ax.add_patch(t1)
        plt.scatter(coords[circles_mask][:, 0], coords[circles_mask][:, 1], s=150, c='green', marker="2",
                    linewidth=1, label='Random circles')
        c = plt.Circle((circle_centers[0][0], circle_centers[0][1]), radius=cicles_radius, color='green', linestyle='--', fill=False)
        ax.add_patch(c)
        c2 = plt.Circle((circle_centers[1][0], circle_centers[1][1]), radius=cicles_radius, color='green', linestyle='--', fill=False)
        ax.add_patch(c2)
        c3 = plt.Circle((circle_centers[2][0], circle_centers[2][1]), radius=cicles_radius, color='lightgreen', linestyle='--', fill=False)
        ax.add_patch(c3)
        ax.axis('equal')
        plt.legend()
        return fig, ax

if __name__ == "__main__":
    # example usage to plot one layout with all the different shapes
    config = Box.from_yaml(filename='config.yml', Loader=yaml.FullLoader)
    layout_generator = LayoutGenerator(**config.turbine_settings)
    layout1 = layout_generator.generate_random_layout(n_points=100, farm_lw_ratio=2, min_dist=5)
    layout2 = layout_generator.generate_random_layout(n_points=30, farm_lw_ratio=0.5, min_dist=3)
    layout_generator.plot(layout1)
    layout_generator.plot(layout2)
    plt.show()
    
    # to generate multiple random layouts ready for processing in PyWake
    layouts = layout_generator.generate_layouts(num_layouts=5)
    print(layouts)