import cv2
import torch
import numpy as np


DEVICE = "cpu"
DTYPE  = torch.float32


def tensor(x, batch_size=None, requires_grad=False, dtype=None, device=None):
    dtype, device = dtype or DTYPE, device or DEVICE
    if batch_size is None:
        return torch.tensor(x, requires_grad=requires_grad, dtype=dtype, device=device)
    else:
        res  = torch.tensor(x, requires_grad=requires_grad, dtype=dtype, device=device).unsqueeze(0)
        return res.repeat(batch_size, *((len(res.shape) - 1) * [1]))


def nparray(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()


def check_shape(x, shape):
    assert len(x.shape) == len(shape), \
        f"dimension mismatch, shapes are:{x.shape} and {shape}"
    for i in range(len(x.shape)):
        if shape[i] is not None:
            assert x.shape[i] == shape[i], \
                f"wrong shape at dim={i}, shapes are:{x.shape} and {shape}"


def canvas_coord(x, unit, orig):
    return int(x*unit)+orig


def render(
    objects, img=None, text=None, text_size=2,
    batch_idxs=[0], canvas_size=8, unit=128, orig=None):

    canvas_size = int(canvas_size + 2)
    xmin = -(canvas_size // 2)
    xmax = (canvas_size // 2)

    if orig is None:
        orig = unit * canvas_size // 2

    if img is None:
        img = np.ones((unit*canvas_size, unit*canvas_size, 3), dtype=np.uint8) * 100

    # paint grid patterns
    for i in range(xmin, xmax + 1):
        for j in range(xmin, xmax + 1):
            color = {0: (220, 220, 220), 1: (150, 150, 150)}[(i + j) % 2]
            ptx, pty = canvas_coord(i, unit, orig), canvas_coord(j, unit, orig)
            img = cv2.rectangle(img, (ptx, pty), (ptx + unit, pty + unit), color, -1)

    # bounding box lines
    bound_lines = [[(xmin + 1, xmin + 1), (xmin + 1, xmax - 1)], [(xmin + 1, xmax - 1), (xmax - 1, xmax - 1)], [(xmax - 1, xmax - 1), (xmax - 1, xmin + 1)], [(xmax - 1, xmin + 1), (xmin + 1, xmin + 1)]]

    # paint bounding box
    for (xstart, ystart), (xend, yend) in bound_lines:
        img = cv2.line(
            img,
            (canvas_coord(xstart, unit, orig), canvas_coord(ystart, unit, orig)),
            (canvas_coord(xend, unit, orig), canvas_coord(yend, unit, orig)), 
            (0, 0, 0), 2, -1
        )

    # paint all objects
    for obj in objects:
        if obj is not None:
            img = obj.render(img, unit, orig, batch_idxs)

    # paint origin
    img = cv2.circle(
        img, (canvas_coord(0, unit, orig), canvas_coord(0, unit, orig)), 
        10, (255, 255, 255), -1
    )

    # paint x axis
    img = cv2.line(
        img, 
        (canvas_coord(xmin, unit, orig), canvas_coord(0, unit, orig)),
        (canvas_coord(xmax, unit, orig), canvas_coord(0, unit, orig)), 
        (255, 255, 255), 2, -1
    )

    # paint y axis
    img = cv2.line(
        img, 
        (canvas_coord(0, unit, orig), canvas_coord(xmin, unit, orig)),
        (canvas_coord(0, unit, orig), canvas_coord(xmax, unit, orig)), 
        (255, 255, 255), 2, -1
    )

    img = np.flip(img, axis=0)

    if text is not None:
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (
            canvas_coord( xmin + 0.3, unit, orig),
            canvas_coord(-xmax + 0.3, unit, orig)
        )
        fontScale              = text_size
        fontColor              = (0, 0, 0)
        thickness              = text_size * 2
        lineType               = cv2.LINE_AA
        
        lines = text.split("\n")

        for i, line in enumerate(lines):
            bottomLeftCornerOfText = (
                canvas_coord( xmin + 0.3, unit, orig),
                canvas_coord(-xmax + 0.6 * (i + 1), unit, orig)
            )
            img = cv2.putText(
                img.astype(np.uint8).copy(),
                line, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)

    return img


class Rigid2dShape:

    """ Base class of shapes """

    def __init__(self, state) -> None:
        check_shape(state, self.STATE_SHAPE)

    @property
    def b(self):
        return self.state.shape[0]

    @property
    def n(self):
        return self.state.shape[1]

    @property
    def d(self):
        return self.state.shape[2]



class Circles(Rigid2dShape):

    """
    Circles:
        Circles can be obstacles or movables
        Movable Circles are using batch mode (all obstacles are not using batch mode)

    tensor shape variable name reference:
        b: batch size
        n: number of this shape
        d: number of floats to specify this shape

    tensor shapes:
        Circles.state                           (b, n, d=5)   [x, v, r]
        Circles.x = Circles.state[:, :, 0:2]      (b, n, d=2)   position
        Circles.v = Circles.state[:, :, 2:4]      (b, n, d=2)   velocity
        Circles.r = Circles.state[:, :, 4:5]      (b, n, d=1)   radius
        Circles.m = Circles.state[:, :, 5:6]      (b, n, d=1)   mass
    """

    STATE_SHAPE = [None, None, 6]

    @property
    def x(self):
        return self.state[:, :, 0:2]

    @property
    def v(self):
        return self.state[:, :, 2:4]

    @property
    def r(self):
        return self.state[:, :, 4:5]

    @property
    def m(self):
        return self.state[:, :, 5:6]

    def __init__(self, state: torch.Tensor, colors=None) -> None:
        super().__init__(state)
        self.state = state
        self.colors = colors
        if colors is not None:
            check_shape(colors, [self.state.shape[1], 3])
        else:
            self.colors = np.array([[255, 0, 0]]).repeat(self.b)
        
        # precompute distance of sum of radius of every 2 circle
        self.r_ij = ((
            self.r + self.r.repeat(1, self.n, 1).view(self.b, self.n, self.n)) \
            * (1 - torch.eye(self.n, dtype=torch.long)[None, :, :]).to(DEVICE)).unsqueeze(-1)

        # precompute sum of mass of every 2 circle
        self.m_ij = self.m + self.m.repeat(1, self.n, 1).view(self.b, self.n, self.n)

        # precompute mass ratio (m_ratio_ij[1, i, j] = 2 * m_j / (m_i + m_j))
        self.m_ratio_ij = (2 * self.m.transpose(1, 2) / self.m_ij).unsqueeze(-1)

    def render(self, img, unit, orig, batch_idxs=[0]):
        for i in range(self.n):
            for batch in batch_idxs:
                center = nparray(self.x[batch, i, :] * unit + orig).astype(int)
                radius = nparray(self.r[batch, i, :] * unit).astype(int)[0]
                colors = self.colors[i].tolist()
                img = cv2.circle(img, center, radius, colors, -1)
                img = cv2.circle(img, center, 5, (255, 255, 255), -1)
                velocity = nparray(self.v[batch, i, :] * unit) / 10
                velocity = velocity.astype(int)
                img = cv2.arrowedLine(img, center, center + velocity, (255, 255, 255), 2)
        return img

    def build_state(self, x, v):
        return torch.cat([x, v, self.state[:, :, 4:]], dim=-1)

    def update(self, a, dt, a_f=0):
        # resolve friction
        dv_f = self.resolve_friction(self.v, a_f * dt)
        # resolve action
        self.state = self.build_state(self.x, self.v + a + dv_f)
        # resolve collision
        x, v = self.resolve_collision(self.x, self.v, dt)
        # update state
        self.state = self.build_state(x, v)

    def resolve_friction(self, v, f_dv):
        dv = torch.zeros(v.shape).to(DEVICE)
        v_norm = v.norm(dim=-1, keepdim=True)
        v_unit = v / torch.clamp(v_norm, min=1e-8)
        v_norm = v_norm.squeeze(-1)
        dv[v_norm > f_dv] = -f_dv * v_unit[v_norm > f_dv]
        dv[v_norm <= f_dv] = -1 * v[v_norm <= f_dv]
        return dv
        # return - 0.1 * v

    def resolve_collision(self, x, v, dt):
        dv, dx = self.dv_dx(x, v, dt)
        v = v + dv
        return x + dt * v + dx, v

    def dv_dx(self, x, v, dt):
        # prevent vector norm from too close to 0
        numeric_threashold = 1e-8

        # don't collide with yourself
        voi_ij = torch.ones((self.b, self.n, self.n, 1)).to(DEVICE)
        voi_ij = voi_ij * (1 - torch.eye(self.n).to(DEVICE).unsqueeze(-1))
        # check_shape(voi_ij, [self.b, self.n, self.n, 1])

        # the next state if no collision happens
        p_next_state = self.build_state(x + v * dt, v)

        # diff[b, i, j, :] = [xi - xj, vi - vj, ri - rj, mi - mj]
        diff = p_next_state[:, :, None, :] \
             - p_next_state.repeat(1, self.n, 1).view(self.b, self.n, self.n, self.d)
        
        # x_ij[b, i, j, :]: relative displacement, v_ij[b, i, j, :]: relative velocity
        x_ij, v_ij = diff[:, :, :, 0:2], diff[:, :, :, 2:4]

        # norm_x_ij[b, i, j, 1]: relative distance between center of i and j
        norm_x_ij = x_ij.norm(dim=-1, keepdim=True)

        # if after dt seconds, no contact is made, a collision will not occur
        voi_ij[norm_x_ij - self.r_ij > 0] = 0

        # unit_x_ij[b, i, j, :]: unit vector of relative displacement
        unit_x_ij = x_ij / torch.clamp(norm_x_ij, min=numeric_threashold)

        # proj_v_ij[b, i, j, 1]: projection length of v_ij onto x_ij, proj_v_ij = unit_x_ij dot v_ij
        proj_v_ij = torch.einsum("bijx,bijx->bij", unit_x_ij, v_ij).unsqueeze(-1)

        # if cos( angle between v_ij & x_ij ) is positive, a collision will not occur
        voi_ij[proj_v_ij > 0] = 0

        # delta v
        dv_ij = -self.m_ratio_ij * proj_v_ij * unit_x_ij

        # time of impact
        toi_ij = (norm_x_ij - self.r_ij) / torch.clamp(proj_v_ij, max=-numeric_threashold)

        # delta x
        dx_ij = torch.clamp(toi_ij - dt, max=0) * dv_ij
        
        # accumulate collision deltas
        dv = (voi_ij * dv_ij).sum(-2)
        dx = (voi_ij * dx_ij).sum(-2)

        return (dv, dx)





