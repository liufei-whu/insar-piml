import math
import torch


DEG2RAD = math.pi / 180.0
PI2INV = 1.0 / (2.0 * math.pi)

# Only used to avoid numerical division by exactly zero at singular points.
EPS = 1e-12


def _nonzero(x, eps=EPS):
    """
    Prevent exact zero denominators while preserving sign.

    This version avoids allocating a full eps tensor on every call.
    PyTorch broadcasts the scalar eps automatically.
    """
    return torch.where(
        torch.abs(x) < eps,
        torch.where(
            x < 0,
            -eps,
            eps,
        ),
        x,
    )


def okada_core(
    alp,
    sd,
    cd,
    length,
    width,
    depth,
    X,
    Y,
    SS=0.0,
    DS=0.0,
    TS=0.0,
):
    """
    PyTorch translation of the C function:

        void Okada(...)

    Optimized by vectorizing the four-corner
    Okada / Chinnery summation.

    Parameters
    ----------
    alp
        1 - 2 * nu

    sd, cd
        sin(dip), cos(dip)

    length, width, depth
        Fault geometry.

    X, Y
        Coordinates in the fault-local coordinate system.
        Can be arbitrary-shaped PyTorch tensors.

    SS
        Strike-slip component.

    DS
        Dip-slip component.

    TS
        Tensile-slip component.

    Returns
    -------
    pSS, pDS, pTS
        Each has shape:
            (3, *X.shape)

        Components correspond to the three displacement
        components in the local Okada system.
    """

    # Convert constants/scalars to tensors compatible with X.
    def as_tensor(value):
        if torch.is_tensor(value):
            return value.to(
                dtype=X.dtype,
                device=X.device,
            )

        return torch.as_tensor(
            value,
            dtype=X.dtype,
            device=X.device,
        )

    alp = as_tensor(alp)
    sd = as_tensor(sd)
    cd = as_tensor(cd)

    length = as_tensor(length)
    width = as_tensor(width)
    depth = as_tensor(depth)

    SS = as_tensor(SS)
    DS = as_tensor(DS)
    TS = as_tensor(TS)

    # --------------------------------------------------
    # Common geometry
    # --------------------------------------------------

    sdcd = sd * cd
    sdsd = sd * sd

    depsd = depth * sd
    depcd = depth * cd

    p = Y * cd + depsd
    q = Y * sd - depcd

    # --------------------------------------------------
    # Vectorized four-corner Okada / Chinnery summation
    #
    # Original combinations:
    #
    # k=0,j=0 -> (length, width), sign +
    # k=0,j=1 -> (0,      width), sign -
    # k=1,j=0 -> (length, 0),     sign -
    # k=1,j=1 -> (0,      0),     sign +
    # --------------------------------------------------

    # Batch-safe four-corner Okada / Chinnery geometry.
    # Scalar sources and batched sources use the same equations.
    # Source parameters are expanded only with trailing singleton
    # dimensions so they broadcast over the observation grid.
    def grid_param(value):
        if value.ndim < X.ndim:
            value = value.reshape(value.shape + (1,) * (X.ndim - value.ndim))
        return value

    length_corner = grid_param(length)
    width_corner = grid_param(width)

    ala = torch.stack(
        [
            length_corner,
            torch.zeros_like(length_corner),
            length_corner,
            torch.zeros_like(length_corner),
        ],
        dim=0,
    )

    awa = torch.stack(
        [
            width_corner,
            width_corner,
            torch.zeros_like(width_corner),
            torch.zeros_like(width_corner),
        ],
        dim=0,
    )

    sign = torch.tensor(
        [
            PI2INV,
            -PI2INV,
            -PI2INV,
            PI2INV,
        ],
        dtype=X.dtype,
        device=X.device,
    ).reshape((4,) + (1,) * X.ndim)

    Xc = X.unsqueeze(0)
    pc = p.unsqueeze(0)
    qc = q.unsqueeze(0)

    xi = Xc - ala
    et = pc - awa

    xi2 = xi * xi
    et2 = et * et
    q2 = qc * qc

    r2 = xi2 + et2 + q2
    r = torch.sqrt(r2)

    d = et * sd - qc * cd
    y = et * cd + qc * sd

    # ------------------------------------------
    # ret = r + et
    # ------------------------------------------

    ret = torch.clamp(
        r + et,
        min=0.0,
    )

    rd = r + d

    # ------------------------------------------
    # tt
    # ------------------------------------------

    q_nonzero = torch.abs(qc) > EPS

    denominator = _nonzero(
        qc * r
    )

    tt_calc = torch.atan(
        xi * et / denominator
    )

    tt = torch.where(
        q_nonzero,
        tt_calc,
        torch.zeros_like(tt_calc),
    )

    # ------------------------------------------
    # re and dle
    # ------------------------------------------

    ret_nonzero = ret > EPS

    re = torch.where(
        ret_nonzero,
        1.0 / _nonzero(ret),
        torch.zeros_like(ret),
    )

    dle_positive = torch.log(
        torch.clamp(
            ret,
            min=EPS,
        )
    )

    dle_zero = -torch.log(
        torch.clamp(
            r - et,
            min=EPS,
        )
    )

    dle = torch.where(
        ret_nonzero,
        dle_positive,
        dle_zero,
    )

    # ------------------------------------------
    # Common quantities
    # ------------------------------------------

    rrx = 1.0 / _nonzero(
        r * (r + xi)
    )

    rre = re / _nonzero(r)

    # ------------------------------------------
    # a1, a3, a4, a5
    # ------------------------------------------

    # C has a special analytical branch for
    # cd == 0, i.e. vertical fault.

    vertical = torch.abs(cd) < EPS

    # ----- vertical-fault branch -----

    rd_safe = _nonzero(rd)
    rd2_safe = _nonzero(rd * rd)

    a1_vertical = (
        -alp / 2.0
        * xi * qc
        / rd2_safe
    )

    a3_vertical = (
        alp / 2.0
        * (
            et / rd_safe
            + y * qc / rd2_safe
            - dle
        )
    )

    a4_vertical = (
        -alp * qc / rd_safe
    )

    a5_vertical = (
        -alp * xi * sd / rd_safe
    )

    # ----- general branch -----

    cd_safe = _nonzero(cd)

    td = sd / cd_safe

    x = torch.sqrt(
        xi2 + q2
    )

    a5_numerator = (
        et * (x + qc * cd)
        + x * (r + x) * sd
    )

    a5_denominator = (
        xi * (r + x) * cd
    )

    a5_calc = (
        alp
        * 2.0
        / cd_safe
        * torch.atan(
            a5_numerator
            / _nonzero(a5_denominator)
        )
    )

    # C:
    #
    # if (xi == 0)
    #     a5 = 0

    a5_general = torch.where(
        torch.abs(xi) < EPS,
        torch.zeros_like(a5_calc),
        a5_calc,
    )

    a4_general = (
        alp
        / cd_safe
        * (
            torch.log(
                torch.clamp(
                    rd,
                    min=EPS,
                )
            )
            - sd * dle
        )
    )

    a3_general = (
        alp
        * (
            y
            / _nonzero(rd)
            / cd_safe
            - dle
        )
        + td * a4_general
    )

    a1_general = (
        -alp
        / cd_safe
        * xi
        / _nonzero(rd)
        - td * a5_general
    )

    # Select the same analytical branch as C.

    a1 = torch.where(
        vertical,
        a1_vertical,
        a1_general,
    )

    a3 = torch.where(
        vertical,
        a3_vertical,
        a3_general,
    )

    a4 = torch.where(
        vertical,
        a4_vertical,
        a4_general,
    )

    a5 = torch.where(
        vertical,
        a5_vertical,
        a5_general,
    )

    # C:
    #
    # a2 = -alp*dle - a3

    a2 = -alp * dle - a3

    req = rre * qc
    rxq = rrx * qc

    # ------------------------------------------
    # Strike slip
    # ------------------------------------------

    mult_ss = sign * SS

    ss0 = -mult_ss * (
        req * xi
        + tt
        + a1 * sd
    )

    ss1 = -mult_ss * (
        req * y
        + qc * cd * re
        + a2 * sd
    )

    ss2 = -mult_ss * (
        req * d
        + qc * sd * re
        + a4 * sd
    )

    # ------------------------------------------
    # Dip slip
    # ------------------------------------------

    mult_ds = sign * DS

    ds0 = -mult_ds * (
        qc / _nonzero(r)
        - a3 * sdcd
    )

    ds1 = -mult_ds * (
        y * rxq
        + cd * tt
        - a1 * sdcd
    )

    ds2 = -mult_ds * (
        d * rxq
        + sd * tt
        - a5 * sdcd
    )

    # ------------------------------------------
    # Tensile slip
    # ------------------------------------------

    mult_ts = sign * TS

    ts0 = mult_ts * (
        q2 * rre
        - a3 * sdsd
    )

    temp = (
        xi * qc * rre
        - tt
    )

    ts1 = mult_ts * (
        -d * rxq
        - sd * temp
        - a1 * sdsd
    )

    ts2 = mult_ts * (
        y * rxq
        + cd * temp
        - a5 * sdsd
    )

    # --------------------------------------------------
    # Chinnery summation over the 4-corner dimension.
    # --------------------------------------------------

    pSS = torch.stack(
        [
            ss0.sum(dim=0),
            ss1.sum(dim=0),
            ss2.sum(dim=0),
        ],
        dim=0,
    )

    pDS = torch.stack(
        [
            ds0.sum(dim=0),
            ds1.sum(dim=0),
            ds2.sum(dim=0),
        ],
        dim=0,
    )

    pTS = torch.stack(
        [
            ts0.sum(dim=0),
            ts1.sum(dim=0),
            ts2.sum(dim=0),
        ],
        dim=0,
    )

    return pSS, pDS, pTS


def disloc(
    x_grid,
    y_grid,
    length,
    width,
    depth,
    dip,
    strike,
    x0,
    y0,
    strike_slip=0.0,
    dip_slip=0.0,
    tensile_slip=0.0,
    poisson=0.25,
):
    """
    PyTorch translation of the single-dislocation case
    of the C function Disloc().

    x_grid, y_grid
        Observation coordinates.

    length, width, depth
        Correspond to pModel[0:3].

    dip
        pModel[3], degrees.

    strike
        pModel[4], degrees.

    x0, y0
        pModel[5:7].

    strike_slip
        pModel[7].

    dip_slip
        pModel[8].

    tensile_slip
        pModel[9].

    Returns
    -------
    ux, uy, uz
        Surface displacement in the original coordinate
        system of x_grid/y_grid.
    """

    device = x_grid.device
    dtype = x_grid.dtype

    def as_tensor(value):
        if torch.is_tensor(value):
            return value.to(
                device=device,
                dtype=dtype,
            )

        return torch.as_tensor(
            value,
            device=device,
            dtype=dtype,
        )

    length = as_tensor(length)
    width = as_tensor(width)
    depth = as_tensor(depth)

    dip = as_tensor(dip)
    strike = as_tensor(strike)

    x0 = as_tensor(x0)
    y0 = as_tensor(y0)

    strike_slip = as_tensor(strike_slip)
    dip_slip = as_tensor(dip_slip)
    tensile_slip = as_tensor(tensile_slip)

    nu = as_tensor(poisson)

    # -----------------------------------------------
    # Dip
    # -----------------------------------------------

    dip_rad = dip * DEG2RAD

    cd_raw = torch.cos(dip_rad)
    sd_raw = torch.sin(dip_rad)

    # Reproduce the C behaviour:
    #
    # if (fabs(cd)<1e-4) {
    #     cd = 0;
    #     if (sd>0)
    #         sd=1;
    #     else
    #         sd=0;
    # }

    near_vertical = torch.abs(cd_raw) < 1e-4

    cd = torch.where(
        near_vertical,
        torch.zeros_like(cd_raw),
        cd_raw,
    )

    sd_vertical = torch.where(
        sd_raw > 0,
        torch.ones_like(sd_raw),
        torch.zeros_like(sd_raw),
    )

    sd = torch.where(
        near_vertical,
        sd_vertical,
        sd_raw,
    )

    # -----------------------------------------------
    # Same rotation as C:
    #
    # Angle = -(90 - strike) * DEG2RAD
    # -----------------------------------------------

    angle = -(90.0 - strike) * DEG2RAD

    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    dx = x_grid - x0
    dy = y_grid - y0

    # Exact C transform:
    #
    # X =
    # cosAngle*(coord_x-x0)
    # - sinAngle*(coord_y-y0)
    # + 0.5*length

    X = (
        cos_angle * dx
        - sin_angle * dy
        + 0.5 * length
    )

    # Y =
    # sinAngle*(coord_x-x0)
    # + cosAngle*(coord_y-y0)

    Y = (
        sin_angle * dx
        + cos_angle * dy
    )

    alp = 1.0 - 2.0 * nu

    pSS, pDS, pTS = okada_core(
        alp=alp,
        sd=sd,
        cd=cd,
        length=length,
        width=width,
        depth=depth,
        X=X,
        Y=Y,
        SS=strike_slip,
        DS=dip_slip,
        TS=tensile_slip,
    )

    # Sum the three slip mechanisms in local coordinates.

    local_x = (
        pSS[0]
        + pDS[0]
        + pTS[0]
    )

    local_y = (
        pSS[1]
        + pDS[1]
        + pTS[1]
    )

    uz = (
        pSS[2]
        + pDS[2]
        + pTS[2]
    )

    # -----------------------------------------------
    # Rotate displacement back exactly as C:
    #
    # x = local[0]
    # y = local[1]
    #
    # output[0] =
    # cosAngle*x + sinAngle*y
    #
    # output[1] =
    # -sinAngle*x + cosAngle*y
    # -----------------------------------------------

    ux = (
        cos_angle * local_x
        + sin_angle * local_y
    )

    uy = (
        -sin_angle * local_x
        + cos_angle * local_y
    )

    return ux, uy, uz


def okada_forward(x0,y0,depth,slip,x_grid,y_grid,length=20.0,width=10.0,dip=60.0,strike=0.0,rake=90.0,poisson=0.25,los_vector=None):
    """
    Public parameters:
        x0, y0  = fault centroid
        depth   = fault centroid depth

    Internally converted to the lower/deeper-edge reference used by disloc.c.
    """

    def tensor(v):
        return v.to(dtype=x_grid.dtype,device=x_grid.device) if torch.is_tensor(v) else torch.as_tensor(v,dtype=x_grid.dtype,device=x_grid.device)

    x0,y0,depth,slip = map(tensor,(x0,y0,depth,slip))
    length,width,dip,strike,rake = map(tensor,(length,width,dip,strike,rake))

    dip_rad = dip*DEG2RAD
    strike_rad = strike*DEG2RAD

    half_horizontal = 0.5*width*torch.cos(dip_rad)
    half_vertical = 0.5*width*torch.sin(dip_rad)

    # Centroid -> midpoint of lower/deeper edge expected by disloc.c
    depth_internal = depth+half_vertical
    x0_internal = x0+half_horizontal*torch.cos(strike_rad)
    y0_internal = y0-half_horizontal*torch.sin(strike_rad)

    rake_rad = rake*DEG2RAD
    strike_slip = slip*torch.cos(rake_rad)
    dip_slip = slip*torch.sin(rake_rad)

    ux,uy,uz = disloc(
        x_grid=x_grid,y_grid=y_grid,
        length=length,width=width,depth=depth_internal,
        dip=dip,strike=strike,
        x0=x0_internal,y0=y0_internal,
        strike_slip=strike_slip,dip_slip=dip_slip,
        tensile_slip=0.0,poisson=poisson
    )

    if los_vector is None:
        return ux,uy,uz

    lx,ly,lz = los_vector
    return lx*ux+ly*uy+lz*uz