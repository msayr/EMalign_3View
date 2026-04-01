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
_ALLOWED_GRIDS_BY_STACK = {}


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


def _get_matching_grids_from_config(config_path, target_resolution):
    """Return active grid indices matching target XY resolution for one config."""
    grids_cfg = _read_grids_config(config_path)
    if grids_cfg is None:
        return set()

    try:
        pixel_size = ast.literal_eval(grids_cfg.get('pixel_size', '[]'))
        grid_active = ast.literal_eval(grids_cfg.get('grid_active', '[]'))
        active_tiles = ast.literal_eval(grids_cfg.get('active_tiles', '[]'))
    except (SyntaxError, ValueError):
        logging.warning(f'Failed to parse [grids] section in {config_path}')
        return set()

    matching_grids = set()
    for i, is_active in enumerate(grid_active):
        if not is_active:
            continue
        if i >= len(active_tiles) or len(active_tiles[i]) == 0:
            continue
        if i >= len(pixel_size):
            continue
        if int(round(float(pixel_size[i]))) == int(target_resolution):
            matching_grids.add(i)
    return matching_grids


def get_tileset_resolution(tileset_path, resolution):
    """Return stack path and matching active grid indices for requested resolution."""
    config_files = _find_config_files(tileset_path)
    if not config_files:
        return None

    # Use ALL config files because acquisition settings can change between files.
    matching_grids = set()
    for config_path in config_files:
        matching_grids.update(
            _get_matching_grids_from_config(config_path, target_resolution=resolution[0])
        )

    if not matching_grids:
        return None

    return (tileset_path, matching_grids)


def get_tilesets(main_dir, resolution, dir_pattern, num_workers):
    """Find stack directories with SBEMimage tile+metadata structure."""
    stack_dirs = glob(os.path.join(main_dir, '*', ''))
    _ALLOWED_GRIDS_BY_STACK.clear()

    stack_list = []
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = [tpe.submit(get_tileset_resolution, d, resolution) for d in stack_dirs]

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
            if not dir_pattern or any(p in stack_name for p in dir_pattern):
                _ALLOWED_GRIDS_BY_STACK[os.path.abspath(result[0])] = result[1]
                stack_list.append(result[0])

    return sorted(stack_list)


def include_tile_path(stack_path, tile_path):
    """Whether a tile path belongs to one of the selected grids for this stack."""
    # SBEMimage stores alignable tiles under <stack>/tiles/...
    # Ignore TIFFs from other folders (workspace, overviews, etc.).
    rel_path = os.path.relpath(tile_path, stack_path).replace('\\', '/')
    if not rel_path.startswith('tiles/'):
        return False

    allowed_grids = _ALLOWED_GRIDS_BY_STACK.get(os.path.abspath(stack_path))
    if allowed_grids is None:
        return True

    try:
        grid_idx, _ = parse_yx_pos_from_name(tile_path)
    except ValueError:
        return False

    return grid_idx in allowed_grids


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
