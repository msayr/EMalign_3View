"""
Utilities for finding and reading files produced by SBEMimage.
"""

import ast
import configparser
import logging
import os
import re

from concurrent import futures
from glob import glob
from tqdm import tqdm

FILE_EXT = '.tif'

_GRID_RE = re.compile(r'g(\d+)')
_TILE_RE = re.compile(r't(\d+)')
_SLICE_RE = re.compile(r'_s(\d+)\.')


def _find_config_files(tileset_path):
    """Return all SBEMimage config files for a stack directory."""
    logs_dir = os.path.join(tileset_path, 'meta', 'logs')
    if os.path.isdir(logs_dir):
        return sorted(glob(os.path.join(logs_dir, 'config_*.txt')))

    # Fallback for provided examples where files are stored directly in a folder.
    return sorted(glob(os.path.join(tileset_path, 'config_*.txt')))


def _read_grids_config(config_path):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if 'grids' not in parser:
        return None
    return parser['grids']


def get_tileset_resolution(tileset_path):
    """Read the XY pixel size (nm) from SBEMimage config metadata."""
    config_files = _find_config_files(tileset_path)
    if not config_files:
        return None

    # Prefer the newest config file for interrupted/restarted acquisitions.
    grids_cfg = _read_grids_config(config_files[-1])
    if grids_cfg is None:
        return None

    try:
        pixel_size = ast.literal_eval(grids_cfg.get('pixel_size', '[]'))
        grid_active = ast.literal_eval(grids_cfg.get('grid_active', '[]'))
        active_tiles = ast.literal_eval(grids_cfg.get('active_tiles', '[]'))
    except (SyntaxError, ValueError):
        logging.warning(f'Failed to parse [grids] section in {config_files[-1]}')
        return None

    active_resolutions = []
    for i, is_active in enumerate(grid_active):
        if not is_active:
            continue
        if i >= len(active_tiles) or len(active_tiles[i]) == 0:
            continue
        if i >= len(pixel_size):
            continue
        active_resolutions.append(float(pixel_size[i]))

    if not active_resolutions:
        # Fall back to any listed pixel size when activity metadata is unavailable.
        if pixel_size:
            active_resolutions = [float(pixel_size[0])]
        else:
            return None

    if len(set(active_resolutions)) > 1:
        logging.warning(
            'Detected mixed active pixel sizes in %s (%s); using the first one',
            tileset_path,
            active_resolutions,
        )

    res = int(round(active_resolutions[0]))
    return (tileset_path, (res, res))


def get_tilesets(main_dir, resolution, dir_pattern, num_workers):
    """Find stack directories with SBEMimage tile+metadata structure."""
    stack_dirs = glob(os.path.join(main_dir, '*', ''))

    stack_list = []
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = [tpe.submit(get_tileset_resolution, d) for d in stack_dirs]

        for f in tqdm(
            futures.as_completed(fs),
            total=len(fs),
            desc=f'Looking for resolution: {resolution}',
            leave=False,
        ):
            result = f.result()
            if result is None:
                continue

            stack_name = os.path.basename(os.path.normpath(result[0]))
            if tuple(resolution) != result[1]:
                continue

            if not dir_pattern or any(p in stack_name for p in dir_pattern):
                stack_list.append(result[0])

    return sorted(stack_list)


def parse_yx_pos_from_name(n):
    """Extract (grid_index, tile_index) from SBEMimage tile names/paths."""
    filename = os.path.basename(n)

    g_match = _GRID_RE.search(filename)
    t_match = _TILE_RE.search(filename)

    if g_match is None or t_match is None:
        # Fallback for relative paths such as tiles\g0000\t0001\*.tif
        path_norm = n.replace('\\', '/')
        g_match = g_match or _GRID_RE.search(path_norm)
        t_match = t_match or _TILE_RE.search(path_norm)

    if g_match is None or t_match is None:
        raise ValueError(f'Could not parse grid/tile indices from name: {n}')

    return (int(g_match.group(1)), int(t_match.group(1)))


def parse_slice_from_name(n):
    """Extract z-slice index from SBEMimage tile names."""
    filename = os.path.basename(n)
    s_match = _SLICE_RE.search(filename)
    if s_match is None:
        raise ValueError(f'Could not parse slice index from name: {n}')
    return int(s_match.group(1))
