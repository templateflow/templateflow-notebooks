# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for plotting all templates in the TemplateFlow Archive.
"""
import numpy as np
import nibabel as nb
import templateflow.api as tflow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nilearn.plotting import plot_surf


def plot_all_templates():
    to_plot = tflow.templates()
    mni = [t for t in to_plot if t[:3] == "MNI"]
    others = [t for t in to_plot if t[:3] != "MNI"]
    to_plot = mni + others
    n_templates = len(to_plot)
    n_rows = int(np.ceil(np.sqrt(n_templates)))
    n_cols = int(np.ceil(n_templates / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_rows, 5 * n_cols))
    fig.tight_layout()
    for i, tpl in enumerate(to_plot):
        row_idx = i % n_rows
        col_idx = i // n_rows
        ax = axs[row_idx, col_idx]
        template_tile(tpl, ax=ax, fig=fig)
    return fig


def template_tile(tpl, ax=None, fig=None):
    """
    Plot a tile for a single template.
    """
    if fig is None:
        fig = plt.figure(figsize=(6, 5))
    if ax is None:
        ax = plt.gca()

    if tpl[:3] == "MNI":
        facecolor = "#88BBDD"
        display_name = tpl[3:]
    else:
        facecolor = "#999999"
        display_name = tpl

    patch_outer = patches.FancyBboxPatch(
        (0, 0),
        1.8,
        1.4,
        linewidth=0,
        edgecolor=None,
        facecolor=facecolor,
        boxstyle="round,rounding_size=0.15",
    )
    patch_inner = patches.FancyBboxPatch(
        (0.5, 0.1),
        1.2,
        1.2,
        linewidth=0,
        edgecolor=None,
        facecolor="#000000",
        boxstyle="round,rounding_size=0.12",
    )

    (x_start, y_start, x_dim, y_dim) = ax.get_position().bounds

    axes_kwargs = {}
    if tpl in ("fsaverage", "fsLR"):
        axes_kwargs["projection"] = "3d"
        axes_kwargs["rasterized"] = True

    img_ax = fig.add_axes(
        ((0.19 * x_dim) + x_start, (0.1 * y_dim) + y_start, 0.8 * x_dim, 0.8 * y_dim),
        **axes_kwargs,
    )

    template_view(tpl, img_ax)

    ax.add_patch(patch_outer)
    ax.add_patch(patch_inner)

    # TODO: font size should be scaled to the axis size
    # instead of hard coded
    fontsize = min(30, 500 / len(display_name)) + 5
    ax.annotate(
        text=display_name,
        xy=(0.12, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
        fontfamily="Whitney",
        weight="bold",
        linespacing=0.0001,
    )

    ax.set_xlim(-0.3, 2.1)
    ax.set_ylim(-0.3, 1.7)
    ax.axis("off")


def search_for_images(query, specification, search_key=None, search_vals=None):
    """
    Helper utility for volumetric fetching.
    """
    for spec in specification:
        query_cur = query.copy()
        query_cur.update(spec)
        if search_key is not None:
            for search_val in search_vals:
                search = {search_key: search_val}
                path = tflow.get(**query_cur, **search)
                if path:
                    break
        else:
            path = tflow.get(**query_cur)
        if not isinstance(path, list):
            break
    return path


def search_for_surfaces(query, search_key, search_vals):
    """
    Helper utility for surface fetching.
    """
    if search_key is not None:
        for search_val in search_vals:
            search = {search_key: search_val}
            path = tflow.get(**query, **search)
            if path:
                break

    return path


def get_image_and_mask(tpl):
    """
    Fetch an image and a mask for plotting a template.
    """
    priority_list_vol = ("T1w", "T2w", "T1map", "T2star", "PDw")
    priority_list_surf = ("inflated", "pial", "sphere")
    # TODO: this spec iteration is inflexible and brittle.
    # Consider changing to a more principled system at some point.
    specification = [
        [],
        [("resolution", 1)],
        [("resolution", 1), ("cohort", 1)],
        [("cohort", 1)],
        [("desc", "brain")],
    ]
    img_query = {"template": tpl, "desc": None}
    mask_query = {
        "template": tpl,
        "desc": "brain",
        "hemi": None,
        "space": None,
        "atlas": None,
        "suffix": "mask",
    }
    surf_query = {
        "template": tpl,
        "hemi": "R",
        "density": ["10k", "32k"],
        "desc": None,
        "space": None,
    }

    tpl_img_path = []
    tpl_img_path = search_for_images(
        img_query, specification, "suffix", priority_list_vol
    )
    tpl_mask_path = search_for_images(mask_query, specification)

    if isinstance(tpl_img_path, list):
        tpl_img = search_for_surfaces(surf_query, "suffix", priority_list_surf)
        tpl_mask = "surf"
        return tpl_img, tpl_mask

    if isinstance(tpl_img_path, list):
        raise ValueError(f"Ambiguous or no reference {tpl_img_path}")
    if isinstance(tpl_mask_path, list):
        tpl_mask_path = None
        tpl_mask = None

    tpl_img = nb.load(tpl_img_path).get_fdata()
    if tpl_mask_path is not None:
        tpl_mask = nb.load(tpl_mask_path).get_fdata()
        # mask and template dimension mismatch: skip masking
        if tpl_img.shape != tpl_mask.shape:
            tpl_mask = None

    return tpl_img, tpl_mask


def template_view(tpl, ax):
    """
    Obtain a view on a template.
    """
    tpl_img, tpl_mask = get_image_and_mask(tpl)

    if isinstance(tpl_mask, str) and tpl_mask == "surf":
        ax.set_facecolor("black")
        plot_surf(
            surf_mesh=str(tpl_img),
            hemi="right",
            view="lateral",
            axes=ax,
            engine="matplotlib",
        )
        return
    elif tpl_mask is not None:
        masked = tpl_img * tpl_mask
    else:
        masked = tpl_img
    # some templates, like RESILIENT, have a lot of negative
    # values, so we'll use the absolute value to select a good slice
    slc_idx = np.abs(masked).sum((0, 1)).argmax()
    slc = np.flipud(masked[:, :, slc_idx].T)
    slc = slc[(np.abs(slc).sum(1) > 0), :]
    slc = slc[:, (np.abs(slc).sum(0) > 0)]
    y, x = slc.shape
    if y > x:
        to_plot = np.zeros((y, y))
        start = int(np.floor((y - x) / 2))
        end = start + x
        to_plot[:, start:end] = slc
    elif x > y:
        to_plot = np.zeros((x, x))
        start = int(np.floor((x - y) / 2))
        end = start + y
        to_plot[start:end, :] = slc
    else:
        to_plot = slc

    ax.imshow(to_plot, cmap="bone")
    ax.axis("off")
