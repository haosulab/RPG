# Let's put the utility here
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_o3d_camera(int, image_res):
    import open3d as o3d
    fx, fy = int[0, 0], int[1, 1]
    cx, cy = int[0, 2], int[1, 2]
    w, h = image_res[1], image_res[0]
    cam = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    return cam

def get_int(image_res):
    fov = 0.23
    image_res = image_res
    int = np.array([
        - np.array([2 * fov / image_res[1], 0, -fov - 1e-5,]),
        - np.array([0, 2 * fov / image_res[1], -fov - 1e-5,]),
        [0, 0, 1]
    ])
    return np.linalg.inv(int)

def draw_geometries(objects, mode='human', img_res=(512, 512), int=None, ext=None):
    if mode == 'human':
        import open3d as o3d
        o3d.visualization.draw_geometries(objects)
    elif mode == 'rgb_array':
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=512, height=512, visible=False)
        if isinstance(objects, tuple) or isinstance(objects, list):
            for geom in objects:
                vis.add_geometry(geom)
        else:
            vis.add_geometry(objects)
        ctr = vis.get_view_control()

        if int is None:
            int = get_int(img_res)

        cam_param = get_o3d_camera(int, img_res)
        o3d_cam = o3d.camera.PinholeCameraParameters()
        o3d_cam.intrinsic = cam_param
        if ext is None:
            from tools.utils import lookat
            R, t = lookat([0.5, 0.5, 0.5], 0., 0., 3.)
            ext = np.eye(4); ext[:3, :3] = R; ext[:3, 3] = t
            ext = np.linalg.pinv(ext)
        o3d_cam.extrinsic = ext #self.get_ext()

        ctr.convert_from_pinhole_camera_parameters(o3d_cam, allow_arbitrary=True)
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        image = np.uint8(np.asarray(image) * 255)
    elif mode == 'notebook':
        #from open3d import JVisualizer
        #visualizer = JVisualizer()
        #for i in objects:
        #    visualizer.add_geometry(i)
        #visualizer.show()
        raise NotImplementedError

        
def np2pcd(xyz):
    import open3d as o3d
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def draw_np(np, mode='human', env=None):
    if mode == 'human':
        return draw_geometries([np2pcd(np)], mode=mode)
    else:
        return draw_geometries([np2pcd(np)], mode='rgb_array', ext=env.renderer.get_ext(), int=env.renderer.get_int())
    
def o3d_draw_batch_pointcloud(pcd, mode='human'):
    for i in pcd:
        if mode != 'notebook':
            img = draw_geometries([np2pcd(pcd)], mode=mode)
            if mode == 'plt':
                import matplotlib.pyplot as plt
                plt.imshow(img)
                plt.show()
        else:
            import trimesh
            return trimesh.points.PointCloud(i.xyz.detach().cpu().numpy())

"""
def plt_draw_batch_pointcloud(pcd):
    from .modules.types import Pointcloud
    pcd: Pointcloud
    for i in pcd:
        p = i.xyz.detach().cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p[:, 0], p[:, 1], p[:, 2])
        plt.show()
"""


def env_draw_batch_pointcloud(pcd, env, mode='plt'):
    images = []
    for i in pcd:
        p = i.xyz.detach().cpu().numpy()
        p = (p - 0.5) * 0.2 + 0.5
        env.set_state(env.empty_state(init=p))
        if mode == 'plt':
            env.render('plt')
        else:
            images.append(env.render('rgb_array'))
    return images

    
from tools.utils import tonumpy
def plot_point_values(v, points, **kwargs):
    points = tonumpy(points)
    points = points.reshape(-1, 2)

    kwargs['vmin']= v.min()
    kwargs['vmax'] = v.max()
    cm = plt.cm.get_cmap('RdYlBu')

    colors = v.reshape(-1)
    sc = plt.scatter(points[:, 0], points[:, 1], c=colors, cmap=cm, **kwargs)
    plt.colorbar(sc)


from tools.utils import tonumpy
def plot_grid_point_values(anchors, values, normalize=True):
    plt.clf()
    anchors = tonumpy(anchors)
    values = tonumpy(values)
    x = anchors[:, 1]
    y = anchors[:, 0]

    min_x = x.min()
    min_y = y.min()
    max_x = x.max() + 1
    max_y = y.max() + 1

    vmin= values.min()
    vmax = values.max()
    import matplotlib
    cm = plt.cm.get_cmap('RdYlBu')
    if not normalize:
        vmin = 0.
        vmax = 1.

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    maps = np.zeros((max_x - min_x, max_y - min_y, 3))
    values = cm((values - vmin) / (vmax - vmin + 1e-9))
    for i in range(len(anchors)):
        maps[x[i] - min_x, y[i] - min_y] = values[i][:3]
    ax = plt.imshow(maps) 
    plt.xlim(-0.5, max_y - min_y-0.5)
    plt.ylim(-0.5, max_x - min_x-0.5)
    cbaxes = plt.gcf().add_axes([0.85, 0.1, 0.03, 0.8]) 
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cm), cax=cbaxes)

    
def plot_colored_embedding(z, points, ax=None, **kwargs):
    # TODO:
    colors = np.array(
        ['b', 'g', 'r', 'c', 'm', 'y', 'C2', 'C3',
            'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0'] * 10,
    )

    points = tonumpy(points)
    points = points.reshape(-1, 2)

    if z.dtype == torch.int32 or z.dtype == torch.int64:
        z = tonumpy(z)
        z = z.reshape(-1)
        label_color = colors[z]
    else:
        z = tonumpy(z)
        label_color = embedding2rgb(z).reshape(-1, 3)
    if ax is None:
        plt.scatter(points[:, 0], points[:, 1], c=label_color, **kwargs)
    else:
        ax.scatter(points[:, 0], points[:, 1], c=label_color, **kwargs)


def embedding2rgb(embedding: torch.Tensor):
    # project the z into different colors ..
    from tools.utils import tonumpy
    from sklearn.decomposition import PCA
    if embedding.shape[-1] < 3:
        components = np.zeros((*embedding.shape[:-1], 3))
        embeddings = (embedding - embedding.min()) / (embedding.max() - embedding.min())
        components[..., :embedding.shape[-1]] = tonumpy(embeddings)
        return components
    if embedding.shape[-1] == 3:
        return embedding
        #return (embedding - embedding.min(axis=0)) / (embedding.max(axis=0) - embedding.min(axis=0))

    pca = PCA(n_components=3) # to 3 dimension

    embedding = tonumpy(embedding)
    shape = embedding.shape
    embedding = embedding.reshape(-1, shape[-1])
    components = pca.fit_transform(embedding)

    #normalize = lambda x: (x - np.min(x)) * 2 / (np.max(x) - np.min(x)) - 1
    #p[:, 0] = normalize(p[:, 0]) * ratio2d
    #p[:, 2] = normalize(p[:, 2]) * ratio2d
    #p[:, 1] = normalize(p[:, 1]) * h
    components = (components - np.min(components, axis=0)
                  )/(np.max(components, axis=0) - np.min(components, axis=0) + 1e-8)

    components = components * (1. - 0.2) + 0.2
    components = components.reshape(*shape[:-1], 3)
    return components
