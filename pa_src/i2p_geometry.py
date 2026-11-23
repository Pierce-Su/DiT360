"""CPU-only pinhole/ERP geometry. x right, y down, z forward; pixel centers.

K indexes source centers [0,W-1] x [0,H-1]. Samples outside that rectangle
are unknown (no edge extension). ERP sampling wraps longitude, clamps latitude.
"""
import numpy as np
from PIL import Image


def intrinsics(width, height, hfov_deg):
    if min(width, height) < 2 or not np.isfinite(hfov_deg) or not 0 < hfov_deg < 180:
        raise ValueError('invalid pinhole dimensions/FoV')
    f = width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    return np.array([[f, 0, (width-1)/2], [0, f, (height-1)/2], [0, 0, 1.]])


def validate(K, R):
    K, R = np.asarray(K, dtype=float), np.asarray(R, dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1]) or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose([K[0, 1], K[1, 0]], 0):
        raise ValueError('K must be finite positive-focal zero-skew pinhole K')
    if R.shape != (3, 3) or not np.isfinite(R).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1):
        raise ValueError('R must be a proper rotation')
    return K, R


def rotation(yaw_deg=0, pitch_deg=0):
    """Positive yaw looks right; positive pitch looks up."""
    y, p = np.deg2rad([yaw_deg, pitch_deg])
    return np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]) @ np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])


def bilinear(rgb, x, y, periodic=False):
    h, w = rgb.shape[:2]
    x = x % w if periodic else np.clip(x, 0, w-1)
    y = np.clip(y, 0, h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1 = (x0+1) % w if periodic else np.minimum(x0+1, w-1)
    y1 = np.minimum(y0+1, h-1)
    a, b = (x-x0)[..., None], (y-y0)[..., None]
    return (1-b)*((1-a)*rgb[y0, x0]+a*rgb[y0, x1]) + b*((1-a)*rgb[y1, x0]+a*rgb[y1, x1])


def perspective_to_sparse_erp(rgb, K, R=None, erp_hw=(1024, 2048), unknown_rgb=(128, 128, 128)):
    K, R = validate(K, np.eye(3) if R is None else R)
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError('RGB source required')
    h, w = erp_hw
    if h < 2 or w != 2*h or len(unknown_rgb) != 3 or not all(0 <= c <= 255 for c in unknown_rgb):
        raise ValueError('2:1 ERP and RGB fill required')
    lon = 2*np.pi*((np.arange(w)+.5)/w-.5)
    lat = np.pi*(.5-(np.arange(h)+.5)/h)
    rays = np.stack(np.broadcast_arrays(np.cos(lat)[:, None]*np.sin(lon), -np.sin(lat)[:, None], np.cos(lat)[:, None]*np.cos(lon)), axis=-1) @ R
    z = rays[..., 2]
    denom = np.where(z > 0, z, 1)
    x = K[0, 0]*rays[..., 0]/denom+K[0, 2]
    y = K[1, 1]*rays[..., 1]/denom+K[1, 2]
    known = (z > 0) & (x >= 0) & (x <= rgb.shape[1]-1) & (y >= 0) & (y <= rgb.shape[0]-1)
    if not known.any():
        raise ValueError('empty footprint')
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:] = unknown_rgb
    canvas[known] = np.rint(bilinear(rgb, x[known], y[known])).clip(0, 255).astype(np.uint8)
    return canvas, known


def erp_to_perspective(rgb, K, R=None, hw=(512, 512)):
    K, R = validate(K, np.eye(3) if R is None else R)
    h, w = hw
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack([(xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones((h, w))], axis=-1) @ R.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    lon, lat = np.arctan2(rays[..., 0], rays[..., 2]), np.arcsin(np.clip(-rays[..., 1], -1, 1))
    eh, ew = rgb.shape[:2]
    return np.rint(bilinear(rgb, (lon/(2*np.pi)+.5)*ew-.5, (.5-lat/np.pi)*eh-.5, periodic=True)).clip(0, 255).astype(np.uint8)


def padded_known_mask(known, scale=8):
    """Released nearest policy, but circular columns match latent [last,body,first]."""
    h, w = known.shape
    if scale != 8 or h % 16 or w % 16:
        raise ValueError('native packing requires VAE scale8 and multiples of16')
    body = np.asarray(Image.fromarray(known.astype(np.uint8)*255).resize((w//16, h//16), Image.Resampling.NEAREST)) == 255
    return np.concatenate([body[:, -1:], body, body[:, :1]], axis=1).reshape(-1, 1).astype(np.float32)
