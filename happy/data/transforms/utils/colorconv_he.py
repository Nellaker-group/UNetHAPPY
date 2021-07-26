import numpy as np
from scipy import linalg
from skimage.util import dtype

# TODO: compare implementation with https://github.com/DIAGNijmegen/pathology-he-auto-augment/blob/main/he-randaugment/custom_hed_transform.py

rgb_from_he_1 = np.array([
    [0.644211, 0.716556, 0.266844],
    [0.092789, 0.954111, 0.283111],
    [0, 0, 0],
])
rgb_from_he_1[2, :] = np.cross(rgb_from_he_1[0, :], rgb_from_he_1[1, :])
he_from_rgb_1 = linalg.inv(rgb_from_he_1)

rgb_from_he_2 = np.array([
    [0.49015734, 0.76897085, 0.41040173],
    [0.04615336, 0.8420684, 0.5373925],
    [0, 0, 0],
])
rgb_from_he_2[2, :] = np.cross(rgb_from_he_2[0, :], rgb_from_he_2[1, :])
he_from_rgb_2 = linalg.inv(rgb_from_he_2)


def rgb2he(rgb):
    """RGB to Haematoxylin-Eosin (HE) color space conversion.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.

    Returns
    -------
    out : ndarray
        The image in HE format, in a 3-D array of shape ``(.., .., 3)``.

    Raises
    ------
    ValueError
        If `rgb` is not a 3-D array of shape ``(.., .., 3)``.


    References
    ----------
    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
           staining by color deconvolution.," Analytical and quantitative
           cytology and histology / the International Academy of Cytology [and]
           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2hed
    >>> ihc = data.immunohistochemistry()
    >>> ihc_he = rgb2he(ihc)
    """
    return mod_separate_stains(rgb, he_from_rgb_1)


def he2rgb(he):
    """Haematoxylin-Eosin-DAB (HED) to RGB color space conversion.

    Parameters
    ----------
    he : array_like
        The image in the HE color space, in a 3-D array of shape
        ``(.., .., 3)``.

    Returns
    -------
    out : ndarray
        The image in RGB, in a 3-D array of shape ``(.., .., 3)``.

    Raises
    ------
    ValueError
        If `hed` is not a 3-D array of shape ``(.., .., 3)``.

    References
    ----------
    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
           staining by color deconvolution.," Analytical and quantitative
           cytology and histology / the International Academy of Cytology [and]
           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2hed, hed2rgb
    >>> ihc = data.immunohistochemistry()
    >>> ihc_he = rgb2he(ihc)
    >>> ihc_rgb = he2rgb(ihc_he)
    """
    return mod_combine_stains(he, rgb_from_he_1)


def mod_separate_stains(rgb, conv_matrix):
    """RGB to stain color space conversion.

    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    conv_matrix: ndarray
        The stain separation matrix as described by G. Landini [1]_.

    Returns
    -------
    out : ndarray
        The image in stain color space, in a 3-D array of shape
        ``(.., .., 3)``.

    Raises
    ------
    ValueError
        If `rgb` is not a 3-D array of shape ``(.., .., 3)``.

    Notes
    -----
    Stain separation matrices available in the ``color`` module and their
    respective colorspace:

    * ``hed_from_rgb``: Hematoxylin + Eosin + DAB
    * ``hdx_from_rgb``: Hematoxylin + DAB
    * ``fgx_from_rgb``: Feulgen + Light Green
    * ``bex_from_rgb``: Giemsa stain : Methyl Blue + Eosin
    * ``rbd_from_rgb``: FastRed + FastBlue +  DAB
    * ``gdx_from_rgb``: Methyl Green + DAB
    * ``hax_from_rgb``: Hematoxylin + AEC
    * ``bro_from_rgb``: Blue matrix Anilline Blue + Red matrix Azocarmine\
                        + Orange matrix Orange-G
    * ``bpx_from_rgb``: Methyl Blue + Ponceau Fuchsin
    * ``ahx_from_rgb``: Alcian Blue + Hematoxylin
    * ``hpx_from_rgb``: Hematoxylin + PAS

    References
    ----------
    .. [1] http://www.dentistry.bham.ac.uk/landinig/software/cdeconv/cdeconv.html

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import separate_stains, hdx_from_rgb
    >>> ihc = data.immunohistochemistry()
    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    """
    rgb = dtype.img_as_float(rgb, force_copy=True)
    rgb += .00001  # make sure no zeros
    stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), conv_matrix)
    return np.reshape(stains, rgb.shape)


def mod_combine_stains(stains, conv_matrix):
    """Stain to RGB color space conversion.

    Parameters
    ----------
    stains : array_like
        The image in stain color space, in a 3-D array of shape
        ``(.., .., 3)``.
    conv_matrix: ndarray
        The stain separation matrix as described by G. Landini [1]_.

    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.

    Raises
    ------
    ValueError
        If `stains` is not a 3-D array of shape ``(.., .., 3)``.

    Notes
    -----
    Stain combination matrices available in the ``color`` module and their
    respective colorspace:

    * ``rgb_from_hed``: Hematoxylin + Eosin + DAB
    * ``rgb_from_hdx``: Hematoxylin + DAB
    * ``rgb_from_fgx``: Feulgen + Light Green
    * ``rgb_from_bex``: Giemsa stain : Methyl Blue + Eosin
    * ``rgb_from_rbd``: FastRed + FastBlue +  DAB
    * ``rgb_from_gdx``: Methyl Green + DAB
    * ``rgb_from_hax``: Hematoxylin + AEC
    * ``rgb_from_bro``: Blue matrix Anilline Blue + Red matrix Azocarmine\
                        + Orange matrix Orange-G
    * ``rgb_from_bpx``: Methyl Blue + Ponceau Fuchsin
    * ``rgb_from_ahx``: Alcian Blue + Hematoxylin
    * ``rgb_from_hpx``: Hematoxylin + PAS

    References
    ----------
    .. [1] http://www.dentistry.bham.ac.uk/landinig/software/cdeconv/cdeconv.html


    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import (separate_stains, combine_stains,
    ...							   hdx_from_rgb, rgb_from_hdx)
    >>> ihc = data.immunohistochemistry()
    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    >>> ihc_rgb = combine_stains(ihc_hdx, rgb_from_hdx)
    """

    stains = dtype.img_as_float(stains)
    logrgb2 = np.dot(-np.reshape(stains, (-1, 3)), conv_matrix)
    rgb2 = np.exp(logrgb2)
    np.clip(rgb2, 0.0, 255.0, out=rgb2)
    return np.reshape(rgb2, stains.shape)
