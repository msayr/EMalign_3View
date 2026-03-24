import cv2
import numpy as np
from tifffile import TiffFile

from emalign.io.process.img_proc import process_image


def load_tif(tif_path, scale=1, process_scheme={}, compute_mask=False):
    '''
    Load tif file.
    ToDo: add support for reading and saving metadata so it could be used for speeding up susbsequent steps.
    '''

    tif = TiffFile(tif_path)
    img = tif.asarray()
    
    if img is None:
        return None, None, None

    img = np.array(img)

    # Process image
    img, mask = process_image(img, process_scheme, compute_mask)

    # Downsample
    if scale < 1:
        return img, cv2.resize(img, None, fx=scale, fy=scale), mask

    return img, None, mask


def load_tilemap(tile_map_paths, invert, process_scheme, scale, skip_missing=False):
    
    '''
    Load a tile map based on provided paths. Apply image processing if specified.
    '''

    z, tile_map_paths= list(tile_map_paths.items())[0]

    if not isinstance(invert, dict):
        invert = dict(zip(list(tile_map_paths.keys()), [invert]*len(tile_map_paths)))
    
    tile_map = {}
    tile_map_ds = {}
    for yx_pos, tile_path in tile_map_paths.items():
        if invert[yx_pos]:
            proc = process_scheme.copy()
            proc['invert'] = True
        else:
            proc = {k:v for k,v in process_scheme.items() if k != 'invert'}

        try:
            img, img_ds, _ = load_tif(tile_path, scale=scale, process_scheme=proc, compute_mask=False)
        except Exception as e:
            if skip_missing:
                img, img_ds = None, None
            else:
                raise e
        tile_map[yx_pos] = img
        tile_map_ds[yx_pos] = img_ds

    return z, tile_map, tile_map_ds