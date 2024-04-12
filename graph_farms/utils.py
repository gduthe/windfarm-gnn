import numpy as np
import torch
from torch_geometric.transforms import FaceToEdge, Polar, KNNGraph, RadiusGraph, Delaunay, Cartesian, LocalCartesian
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_undirected
import math

def rotate(rot_center, coords, angle):
    """ Rotate a set of 2D points counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        
        args:
        rot_center: tuple, the center of the rotation
        coords: (n, 2) np.array, the coordinates of the points to rotate
        angle: float, the angle of the rotation in radians
    """
    # create the rotation matrix
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]])
    
    # translate the point to the origin
    new_coords = np.array(coords) - np.array(rot_center)
    
    # rotate the points around the origin
    new_coords = np.matmul(R, new_coords.T).T
    
    # translate the points back
    new_coords += np.array(rot_center)

    return new_coords


def tri_area(x1, y1, x2, y2, x3, y3):
    """ Calculate the area of a triangle given its vertices.""" 
    a = abs((x1 * (y2 - y3) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) / 2.0)
    return a


def is_inside_triangle(x1, y1, x2, y2, x3, y3, x, y):
    """ Check if a point is inside a triangle. """
    # calculate area of triangle ABC
    A = tri_area(x1, y1, x2, y2, x3, y3)

    # calculate area of triangle PBC
    A1 = tri_area(x, y, x2, y2, x3, y3)

    # calculate area of triangle PAC
    A2 = tri_area(x1, y1, x, y, x3, y3)

    # calculate area of triangle PAB
    A3 = tri_area(x1, y1, x2, y2, x, y)

    # check if sum of A1, A2 and A3 is same as A
    if math.isclose(A, A1 + A2 + A3, abs_tol=1e-1):
        return True
    else:
        return False
    
@torch.jit.script
def ray_circle_intersection_kernel(ray_origins, ray_directions, circle_centers, circle_radii):
    # Calculate the vector from ray origin to circle center
    oc = circle_centers.unsqueeze(0) - ray_origins.unsqueeze(1)

    # Calculate the projection of oc onto the ray direction
    tca = torch.sum(oc * ray_directions.unsqueeze(1), dim=2)

    # Calculate the squared distance from the ray to the circle center
    d2 = torch.sum(oc * oc, dim=2) - tca * tca

    # Calculate the half chord distance
    thc = torch.sqrt(circle_radii.unsqueeze(0) * circle_radii.unsqueeze(0) - d2)

    # Calculate the intersection distances
    t0 = tca - thc
    t1 = tca + thc

    # Check if the rays intersect the circles
    intersections = (t0 >= 0) | (t1 >= 0)

    # Check if the circle is centered on the ray origin
    centered_on_origin = torch.all(oc == 0, dim=2)

    # Ignore intersections if the circle is centered on the ray origin
    intersections = intersections & ~centered_on_origin

    # Create a tensor to store the circle IDs
    circle_ids = torch.arange(circle_centers.size(0), device=circle_centers.device).unsqueeze(0).expand(ray_origins.size(0), -1)

    # Get the IDs of the intersected circles
    intersected_circle_ids = torch.where(intersections, circle_ids, -1)

    return intersections, intersected_circle_ids

def raycasting_connectivity(positions, wd, turbine_diameter, num_rays: int = 100, ray_cast_angle: int=90):
    ''' Function to use raycasting to obtain the connectivity of a wind farm graph

        args:
            positions, (torch.Tensor): (n,2) The positions of the nodes in the wind farm graph
            wd, float: The wind direction in degrees (0 degrees is from the north)
            turbine_diameter, float: The diameter of the wind turbine
            num_rays, int: The number of rays to cast from each node
            ray_cast_angle, int: The angle of the cone in degrees to cast the rays from the wind direction
            
    '''
    device = positions.device
    num_nodes = positions.shape[0]
    circle_centers = positions
    circle_radii = torch.ones(num_nodes, device=device) * turbine_diameter
    
    # Generate equally spaced ray directions between -pi/4 and pi/4 centered around the wind direction
    wd = (270 - wd + 180) % 360
    wd = wd * math.pi / 180 # Convert to radians
    ray_cast_angle = ray_cast_angle * math.pi / 180 # Convert to radians
    angles = torch.linspace(-ray_cast_angle/2, ray_cast_angle/2, num_rays) + wd
    ray_directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    # tile the ray directions to match the number of nodes
    ray_directions = ray_directions.repeat(num_nodes, 1)

    # Expand ray origins and directions for broadcasting
    ray_origins = torch.repeat_interleave(positions, num_rays, dim=0)
    
    # Perform ray-circle intersection on the specified device
    intersections, ids = ray_circle_intersection_kernel(ray_origins.to(device), ray_directions.to(device), circle_centers.to(device), circle_radii.to(device))
    
    # For each node, get the list of unique intersected circle IDs
    receivers = ids.view(num_nodes, num_rays, -1).unique(dim=2).reshape(num_nodes, -1).unique(dim=1)
    
    # get the senders
    senders = torch.repeat_interleave(torch.arange(num_nodes, device=device), receivers.shape[1], dim=0)
    
    # reshape the receivers and get  mask of -1 values
    receivers = receivers.reshape(-1)
    mask = receivers != -1
    
    # create the unidrected edge index
    edge_index = torch.stack([senders[mask], receivers[mask]], dim=0)
    edge_index = to_undirected(edge_index)
    
    return edge_index

def to_graph(points: np.array, connectivity: str, min_dist=None, constant=30, add_edge='polar', wd=None, turbine_diameter=130, ray_cast_angle=90):
    '''
    Converts np.array to torch_geometric.data.data.Data object with the specified connectivity and edge feature type.
    
    args:
    points: np.array, the coordinates of the points to convert
    connectivity: str, the kind of connectivity to use: either delaunay, knn, radial, fully_connected
    min_dist: float, the minimal distance between the points (required for radial connectivity)
    constant: float, the constant to multiply the minimal distance to get the radius of the radial connectivity
    add_edge: str, the kind of geometrical edge feature to use: either polar, cartesian, local cartesian
    wd: float, the wind direction in degrees (0 degrees is from the north) (required for raycasting connectivity)
    '''
    assert connectivity in ['delaunay', 'knn', 'radial', 'fully_connected', 'raycasting']
    assert (points.shape[1] == 2)
    
    t = torch.Tensor(points)
    x = Data(pos=t)
    if connectivity.casefold() == 'delaunay':
        d = Delaunay()
        e = FaceToEdge()
        g = e(d(x))
    elif connectivity.casefold() == 'knn':
        kv = math.ceil(np.sqrt(len(points)))
        knn = KNNGraph(k=kv)
        g = knn(x)
    elif connectivity.casefold() == 'radial':
        if (min_dist is not None):
            radius = min_dist * constant
            r = RadiusGraph(r=radius)
            g = r(x)
        else:
            raise ValueError('Minimal distance between turbines is required.')
    elif connectivity.casefold() == 'fully_connected':
        adj = torch.ones(t.shape[0], t.shape[0])
        g = Data(pos=t, edge_index=dense_to_sparse(adj.fill_diagonal_(0))[0])
    elif connectivity.casefold() == 'raycasting':
        g = Data(pos=t, edge_index=raycasting_connectivity(t, wd, turbine_diameter, num_rays=100, ray_cast_angle=ray_cast_angle))
    else:
        raise ValueError('Please define the connectivity scheme (available types: : \'delaunay\', \'knn\', \'radial\', , \'fully_connected\', \'raycasting\')')

    if add_edge == 'polar'.casefold():
        p = Polar(norm=False)
        p(g)
    elif add_edge == 'cartesian'.casefold():
        c = Cartesian(norm=False)
        c(g)
    elif add_edge == 'local cartesian'.casefold():
        lc = LocalCartesian(norm=False)
        lc(g)
    else:
        raise ValueError(
            'Please select a coordinate system that is supported (available types: : \'polar\', \'cartesian\' or \'local cartesian\')')
    return g

if __name__ == "__main__":
    # test the rotate function
    print(rotate((0,0), np.array([[1,0], [0,1]]), math.pi/2))
    print(rotate((1,1), np.array([[1,0], [0,1]]), math.pi/2))