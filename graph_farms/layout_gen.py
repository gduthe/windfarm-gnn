import numpy as np
from utils import rotate, is_inside
import random
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib

def generate_layout(n_points, length, width, min_dist):

    width_ratio = length / width
    num_y = np.int32(np.sqrt(n_points / width_ratio)) + 1
    num_x = np.int32(n_points / num_y) + 1

    # create regularly spaced points
    x = np.linspace(0., length - 1, num_x, dtype=np.float32)
    y = np.linspace(0., width - 1, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    # compute spacing
    init_dist = np.min((x[1] - x[0], y[1] - y[0]))

    # perturb points
    max_movement = (init_dist - min_dist) / 2

    # create a smaller rectangle
    x1 = np.linspace(0. + max_movement, length - 1 - max_movement, num_x, dtype=np.float32)
    y1 = np.linspace(0. + max_movement, width - 1 - max_movement, num_y, dtype=np.float32)
    coords1 = np.stack(np.meshgrid(x1, y1), -1).reshape(-1, 2)
    init_dist1 = np.min((x1[1] - x1[0], y1[1] - y1[0]))
    max_movement1 = (init_dist1 - min_dist) / 2

    noise = np.random.uniform(low=-max_movement1,
                                high=max_movement1,
                                size=(len(coords1), 2))
    coords1 += noise
    l = [0, np.pi / 8, np.pi / 4, np.pi / 2]
    alpha = random.choice(l)
    coords2 = np.array([rotate((0, 0), c, alpha) for c in coords1])

    ## Ellipse

    theta = np.arange(0, 2 * np.pi, 0.01)

    v = random.choice([2, 4, 6])

    a = length / v
    b = width / 2

    x0, y0 = rotate((0, 0), (length / 2, width / 2), alpha)

    xpos = (a) * np.cos(theta)
    ypos = (b) * np.sin(theta)

    extra = random.choice([np.pi / 8, np.pi / 16, 0, - np.pi / 16, -np.pi / 8])

    new_xpos = x0 + (xpos) * np.cos(-alpha + extra) + (ypos) * np.sin(-alpha + extra)
    new_ypos = y0 + (-xpos) * np.sin(-alpha + extra) + (ypos) * np.cos(-alpha + extra)

    s1 = ((a ** 2) * (np.sin(alpha - extra) ** 2) + (b ** 2) * (np.cos(alpha - extra) ** 2)) * (
                coords2[:, 0] - x0) ** 2
    s2 = 2 * (b ** 2 - a ** 2) * np.sin(alpha - extra) * np.cos(alpha - extra) * (coords2[:, 0] - x0) * (
                coords2[:, 1] - y0)
    s3 = ((a ** 2) * (np.cos(alpha - extra) ** 2) + (b ** 2) * (np.sin(alpha - extra) ** 2)) * (
                coords2[:, 1] - y0) ** 2

    inside = s1 + s2 + s3 < (a ** 2) * (b ** 2)

    c4 = (np.array([new_xpos, new_ypos]))

    ## Triangle

    x11 = length / 2
    y11 = width
    x21 = 0
    y21 = 0
    x31 = length
    y31 = 0
    x = coords2[:, 0]
    y = coords2[:, 1]
    extra = random.choice([np.pi / 8, np.pi / 16, 0, - np.pi / 16, -np.pi / 8])
    x1, y1 = rotate((0, 0), (x11, y11), alpha + extra)
    x2, y2 = rotate((0, 0), (x21, y21), alpha + extra)
    x3, y3 = rotate((0, 0), (x31, y31), alpha + extra)
    m = [is_inside(x1, y1, x2, y2, x3, y3, xp, yp) for xp, yp in zip(x, y)]

    ## small circles

    x = coords2[:n_points][:, 0]
    y = coords2[:n_points][:, 1]

    random_turb = random.choice(coords2[:n_points])
    random_turb2 = random.choice(coords2[:n_points])
    random_turb3 = random.choice(coords2[:n_points])
    radius = length / 4

    mm = (x - random_turb[0]) ** 2 + (y - random_turb[1]) ** 2 < radius ** 2
    mm2 = (x - random_turb2[0]) ** 2 + (y - random_turb2[1]) ** 2 < radius ** 2
    mm3 = (x - random_turb3[0]) ** 2 + (y - random_turb3[1]) ** 2 < radius ** 2

    return (coords2[:n_points], c4, inside[:n_points], alpha, m[:n_points], np.array([[x1, y1], [x2, y2], [x3, y3]]),
    radius, random_turb, mm[:n_points], random_turb2, mm2[:n_points], random_turb3, mm3[:n_points])

def plot(n_points, length, width, min_dist, x=None, y=None):
    """ Plotting function to visualize a random wind farm layout. """
    
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False

    if np.sum(x) is not None and np.sum(y) is not None:
        coords = np.array([[a, b] for a, b in zip(x, y)])
    else:
        coords, c4, i, alpha, m, t, rad, rt, mm, rt2, mm2, rt3, mm3 = generate_layout(n_points, length,
                                                                                                    width, min_dist)
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.scatter(coords[:, 0], coords[:, 1], s=150, c='black', marker="2",
                linewidth=1, label=str(len(coords[:, 0])) + ' WT')
    r = Rectangle((0, 0), length, width, linestyle='--', fill=False)
    t2 = matplotlib.transforms.Affine2D().rotate(alpha) + ax.transData
    r.set_transform(t2)
    plt.scatter(coords[i][:, 0], coords[i][:, 1], s=150, c='red', marker="2",
                linewidth=1, label='Inside ellipse')
    plt.plot(c4[0, :], c4[1, :], 'r', linestyle='--')
    ax.add_patch(r)
    plt.scatter(coords[m][:, 0], coords[m][:, 1], s=150, c='blue', marker="2",
                linewidth=1, label='Inside triangle')
    t1 = plt.Polygon(t[:3, :], color='Blue', linestyle='--', fill=False)
    ax.add_patch(t1)
    plt.scatter(coords[mm][:, 0], coords[mm][:, 1], s=150, c='green', marker="2",
                linewidth=1, label='Random circles')
    c = plt.Circle((rt[0], rt[1]), radius=rad, color='green', linestyle='--', fill=False)
    ax.add_patch(c)
    plt.scatter(coords[mm2][:, 0], coords[mm2][:, 1], s=150, c='green', marker="2",
                linewidth=1)
    c2 = plt.Circle((rt2[0], rt2[1]), radius=rad, color='green', linestyle='--', fill=False)
    ax.add_patch(c2)
    plt.scatter(coords[mm3][:, 0], coords[mm3][:, 1], s=150, c='green', marker="2",
                linewidth=1)
    c3 = plt.Circle((rt3[0], rt3[1]), radius=rad, color='green', linestyle='--', fill=False)
    ax.add_patch(c3)
    ax.axis('equal')
    plt.legend()
    return fig, ax
    plt.show()

if __name__ == "__main__":
    # example usage
    plot(100, 1000, 1000, 100)
    plt.show()