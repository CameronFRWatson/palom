import skimage.util
import skimage.filters
import skimage.exposure
import numpy as np
import warnings

def imagej_rgb2cmyk(rgb_img):
    shape = rgb_img.shape
    assert 3 in shape, (
        'Image of shape {} is not an RGB image'.format(shape)
    )
    channel_idx = shape.index(3)
    rgb_img = skimage.util.img_as_float(rgb_img)
    rgb = np.moveaxis(rgb_img, channel_idx, 0)

    cmy = 1-rgb
    k = cmy.min(axis=0)
   
    s = 1-k

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', 'invalid value encountered in true_divide', RuntimeWarning,
        )
        f_cmy = (cmy-k) / s
    f_cmy *= ~(k >= 1)

    return np.append(f_cmy, k.reshape(1, *k.shape), axis=0)

def cmyk2marker_int(cmyk_img, min, max):
    marker_int = cmyk_img[1] + cmyk_img[2]
    # `selem` kwarg renamed to `footprint` in skimage v0.19
    marker_int = skimage.filters.median(marker_int, np.ones((3,3)))

    marker_int = skimage.exposure.rescale_intensity(
        marker_int, in_range=(min, max), out_range=(0, 1)
    )
    return marker_int

def ohsu_cmyk2marker_int(cmyk_img):
    marker_int = cmyk_img[1] + cmyk_img[2]
    # the latest workflow applys median filter here while earlier
    # workflow seems to apply median filter after rescale_intensity
    marker_int = skimage.filters.median(marker_int, np.ones((3,3)))

    max_int = marker_int.max()
    marker_int = skimage.exposure.rescale_intensity(
        marker_int.astype(float),
        in_range=(0.05*max_int, 0.95*max_int),
        out_range=(0, 1)
    )
    return skimage.util.img_as_ubyte(marker_int)

def rgb2aec(rgb_img):
    cmyk_img = imagej_rgb2cmyk(rgb_img)
    aec = cmyk_img[1] + cmyk_img[2]
    return skimage.filters.median(aec, np.ones((3,3)))

def _ensure_0_255(rgb_img):
    """Convert an RGB image array to float64 in [0, 255] range.

    Handles uint8/uint16 input (preserves 0-255 for uint8, scales
    uint16 down) and float input in [0, 1] or [0, 255].
    """
    if np.issubdtype(rgb_img.dtype, np.integer):
        info = np.iinfo(rgb_img.dtype)
        if info.max == 255:
            return rgb_img.astype(np.float64)
        else:
            return rgb_img.astype(np.float64) / info.max * 255.0
    else:
        rgb_img = rgb_img.astype(np.float64)
        if rgb_img.max() <= 1.0:
            return rgb_img * 255.0
        return rgb_img


def compute_ym_float(rgb_img):
    """Compute the Y+M float signal from an RGB image using the ImageJ
    RGB_to_CMYK formula.

    Y = (max(R,G,B) - B) / max(R,G,B) * 255
    M = (max(R,G,B) - G) / max(R,G,B) * 255

    Parameters
    ----------
    rgb_img : np.ndarray
        RGB image in (H, W, 3) order. Accepts uint8, uint16, or float
        in [0, 1] / [0, 255] range.

    Returns
    -------
    np.ndarray
        Y+M float signal, float64 (H, W), in [0, 510] range.
    """
    rgb_img = _ensure_0_255(rgb_img)
    r = rgb_img[:, :, 0]
    g = rgb_img[:, :, 1]
    b = rgb_img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)

    # np.where evaluates both branches, so (mx-b)/mx produces NaN/inf
    # where mx==0 even though those entries are discarded.  Suppress.
    with np.errstate(divide='ignore', invalid='ignore'):
        Y = np.where(mx > 0, (mx - b) / mx * 255.0, 0.0)
        M = np.where(mx > 0, (mx - g) / mx * 255.0, 0.0)
    return Y + M


def rgb2aec_v2(rgb_img, ym_max=None):
    """AEC CMYK color deconvolution matching ImageJ's RGB_to_CMYK + macro
    pipeline.

    Key differences from :func:`rgb2aec`:
    - CMYK channels computed as Y=(max-B)/max*255, M=(max-G)/max*255
      in 0-255 space (ImageJ macro formula)
    - Y+M addition is float (no clipping)
    - No median filter
    - Float-to-8bit conversion maps [0, max(Y+M)] -> [0, 255] with rounding
    - Contrast stretch: 5%/95% of max (fixed at 12.75/242.25 when
      ym_max is pre-computed from the same image)

    Parameters
    ----------
    rgb_img : np.ndarray
        RGB image in (H, W, 3) order. Accepts uint8, uint16, or float
        in [0, 1] / [0, 255] range.
    ym_max : float, optional
        Pre-computed global max of the Y+M float signal. If None,
        computed from this array. For block-based Dask processing,
        should be pre-computed from the thumbnail level to ensure
        consistent normalization across blocks.

    Returns
    -------
    np.ndarray
        Deconvolved AEC signal, uint8 (H, W), in [0, 255] range.
    """
    ym_float = compute_ym_float(rgb_img)

    if ym_max is None:
        ym_max = ym_float.max()
    if ym_max == 0:
        return np.zeros(ym_float.shape, dtype=np.uint8)

    # Float to 8-bit: map [0, ym_max] -> [0, 255] with rounding
    ym_8bit = np.clip(
        np.round(ym_float / ym_max * 255), 0, 255
    ).astype(np.uint8)

    # Contrast stretch: 5%/95% of global image max
    if ym_8bit.max() == 0:
        return ym_8bit
    newmin = 255 * 0.05
    newmax = 255 * 0.95
    result = skimage.exposure.rescale_intensity(
        ym_8bit, in_range=(newmin, newmax), out_range=(0, 255)
    ).astype(np.uint8)

    return result