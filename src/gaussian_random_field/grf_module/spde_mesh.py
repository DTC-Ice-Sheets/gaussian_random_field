"""SPDE mesh for use with the GRF module."""

from functools import cached_property
from pathlib import Path

import numpy as np
import rpy2.robjects as ro
import shapely
from rpy2.robjects.packages import importr


class SpdeMesh:
    """SPDE mesh for use with the GRF module."""

    meshes_path = Path(__file__).resolve().parent / "meshes"

    def __init__(
        self,
        mesh_name: str,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.01,
    ) -> None:
        importr("fmesher")
        importr("INLA")

        c = ro.r["c"]
        inla_spde2_pcmatern = ro.r["inla.spde2.pcmatern"]
        fm_as_fm = ro.r["fm_as_fm"]

        self.mesh_name = mesh_name
        ro.r.load(str(SpdeMesh.meshes_path / (mesh_name + ".RData")))  # type: ignore
        self.mesh = fm_as_fm(ro.r["mesh1"])  # type: ignore

        self.vertices = np.array(self.mesh.rx2("loc"))[:, :2]
        tv = np.array(self.mesh.rx2("graph").rx2("tv")).flatten()
        self.triangles = []
        self.triangle_verts = []
        for i in range(0, len(tv), 3):
            verts = [tv[idx] - 1 for idx in [i, i + 1, i + 2]]
            self.triangles.append(shapely.Polygon([self.vertices[v] for v in verts]))
            self.triangle_verts.append(verts)

        self.spde = inla_spde2_pcmatern(  # type: ignore
            mesh=self.mesh,  # type: ignore
            alpha=2,
            prior_range=c(50000, 0.5),  # type: ignore
            prior_sigma=c(prior_sigma_1, prior_sigma_2),  # type: ignore
        )

    @cached_property
    def footprint(self) -> shapely.Geometry:
        """The area the mesh covers, as a single polygon.

        Returns
        -------
        shapely.Geometry
            The union of the mesh triangles.
        """
        return shapely.union_all(self.triangles)

    def covers(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Which of the given positions fall on the mesh.

        A position off the mesh cannot be projected onto any triangle, which
        gives an all-zero row in the projector matrix and makes INLA refuse the
        fit. Positions exactly on a triangle edge do project, so the boundary
        counts as covered.

        Parameters
        ----------
        x : np.ndarray
            Eastings, in the mesh's own coordinate reference system.
        y : np.ndarray
            Northings, in the mesh's own coordinate reference system.

        Returns
        -------
        np.ndarray
            Boolean, true where the position falls on the mesh.
        """
        return np.asarray(shapely.intersects_xy(self.footprint, x, y))
