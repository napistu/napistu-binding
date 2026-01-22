"""
DiffDock molecular docking runner with flexible backend support.

Classes
-------
PoseResult
    Container for a predicted binding pose.
DiffDockRunner
    DiffDock molecular docking runner with flexible backend support.
DiffDockManager
    Manager for DiffDock which handles file loading and caching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from napistu.utils.docker_utils import DockerImageManager, ImageInfo
from napistu_torch.ml.hugging_face import HFSpacesClient
from napistu_torch.utils.base_utils import ensure_path

from napistu_binding.load.constants import DIFFDOCK_CONSTANTS
from napistu_binding.load.nmol import nMol
from napistu_binding.load.nstructure import nStructure

logger = logging.getLogger(__name__)


class PoseResult:
    """
    Container for a predicted binding pose.

    Attributes
    ----------
    ligand_coords : np.ndarray
        3D coordinates of ligand atoms, shape (n_atoms, 3)
    confidence_score : float
        Confidence score for this pose (lower is better)
    rank : int
        Ranking of this pose (1 = best)
    protein_coords : Optional[np.ndarray]
        3D coordinates of protein atoms (unchanged from input)
    """

    def __init__(
        self, ligand_coords, confidence_score: float, rank: int, protein_coords=None
    ):
        self.ligand_coords = ligand_coords
        self.confidence_score = confidence_score
        self.rank = rank
        self.protein_coords = protein_coords

    def __repr__(self):
        return (
            f"PoseResult(rank={self.rank}, "
            f"confidence={self.confidence_score:.3f}, "
            f"n_atoms={len(self.ligand_coords)})"
        )


class DiffDockRunner:
    """
    DiffDock molecular docking runner with flexible backend support.

    Supports execution Hugging Face Spaces

    Parameters
    ----------
    backend : str, default="huggingface"
        Execution backend:
        - "huggingface": Use HF Spaces API (works on all platforms)
        - "docker": Local Docker image
    hf_token : Optional[str]
        HuggingFace API token
    space_id : Optional[str]
        HuggingFace Space ID for DiffDock

    Examples
    --------
    >>> # Use official DiffDock Space (recommended for Mac)
    >>> runner = DiffDockRunner(backend="huggingface")
    >>> poses = runner.predict_poses(
    ...     protein="protein.pdb",
    ...     ligand="CCO",  # SMILES string
    ...     num_poses=5
    ... )
    >>>
    >>> # Check best pose
    >>> best = poses[0]
    >>> print(f"Confidence: {best.confidence_score:.2f}")
    >>> print(f"Coordinates shape: {best.ligand_coords.shape}")
    """

    def __init__(
        self,
        backend: str = DIFFDOCK_CONSTANTS.BACKENDS.HUGGINGFACE,
        hf_token: Optional[str] = None,
        space_id: Optional[str] = DIFFDOCK_CONSTANTS.SPACE_ID,
        diffdock_image: Optional[ImageInfo] = None,
    ):
        """Initialize DiffDock runner with specified backend."""
        self.backend = backend
        self.hf_token = hf_token
        self.space_id = space_id
        self.diffdock_image = diffdock_image

        # Validate backend choice
        valid_backends = [
            DIFFDOCK_CONSTANTS.BACKENDS.HUGGINGFACE,
            DIFFDOCK_CONSTANTS.BACKENDS.DOCKER,
        ]
        if backend not in valid_backends:
            raise ValueError(
                f"Invalid backend: {backend}. " f"Choose from: {valid_backends}"
            )

        # Initialize HF Spaces client if needed
        # This inherits all the auth validation from HFClient
        self._hf_client = None
        if backend == DIFFDOCK_CONSTANTS.BACKENDS.HUGGINGFACE:
            self._hf_client = HFSpacesClient(space_id=self.space_id, hf_token=hf_token)
            logger.info("Initialized DiffDock with HF Spaces backend")

    def predict_poses(
        self,
        protein: Union[str, Path],
        ligand: Union[str, Path],
        out_dir: Union[str, Path],
        num_poses: int = 10,
        verbose: bool = True,
        **kwargs,
    ) -> List[PoseResult]:
        """
        Predict binding poses for protein-ligand pair.

        Parameters
        ----------
        protein : str or Path
            Path to PDB file or PDB identifier
        ligand : str or Path
            SMILES string or path to ligand file (SDF, MOL2)
        out_dir : str or Path
            Directory to save results. This will generally be created by the runner.
        num_poses : int, default=10
            Number of poses to generate
        verbose : bool, default=True
            Whether to print verbose output
        **kwargs
            Additional backend-specific parameters

        Returns
        -------
        List[PoseResult]
            List of predicted poses sorted by confidence (best first)

        Examples
        --------
        >>> runner = DiffDockRunner()
        >>> poses = runner.predict_poses("protein.pdb", "CCO", num_poses=5)
        >>>
        >>> # Examine top 3 poses
        >>> for i, pose in enumerate(poses[:3], 1):
        ...     print(f"Pose {i}: confidence={pose.confidence_score:.2f}")
        """
        out_dir = ensure_path(out_dir)
        if self.backend == DIFFDOCK_CONSTANTS.BACKENDS.HUGGINGFACE:
            return self._run_huggingface(
                protein, ligand, out_dir, num_poses, verbose, **kwargs
            )
        if self.backend == DIFFDOCK_CONSTANTS.BACKENDS.DOCKER:
            return self._run_docker(
                protein, ligand, out_dir, num_poses, verbose, **kwargs
            )

    def _run_docker(
        self,
        protein: Union[str, Path],
        ligand: Union[str, Path],
        out_dir: Path,
        num_poses: int,
        verbose: bool,
        **kwargs,
    ) -> List[PoseResult]:
        """
        Run DiffDock via Docker.

        Parameters
        ----------
        protein : Union[str, Path]
            Path to PDB file or PDB identifier
        ligand : Union[str, Path]
            SMILES string or path to ligand file
        out_dir : Path
            Directory to save results
        num_poses : int
            Number of poses to generate
        verbose : bool, default=True
            Whether to print verbose output
        **kwargs
            Additional parameters composed into the command arguments.
            Kwargs are converted to command-line flags (e.g., {"key": "value"} -> ["--key", "value"]).
            Note: Volumes are fixed and not affected by kwargs.
        """

        if self.diffdock_image is None:
            diffdock_image = DIFFDOCK_IMAGE

        # validate that the image is available and a Docker daemon is running
        manager = DockerImageManager(diffdock_image)

        # separate the out_dir into the parent and leaf directories
        out_dir_parent = out_dir.parent
        out_dir_leaf = out_dir.name

        to_be_mounted_volumes = {str(out_dir_parent): "/results:rw"}
        if isinstance(protein, Path):
            protein_string = f"/data/{protein.name}"
            to_be_mounted_volumes[str(protein.parent)] = "/data:ro"
        else:
            protein_string = protein
        if isinstance(ligand, Path):
            ligand_string = f"/data/{ligand.name}"
            to_be_mounted_volumes[str(ligand.parent)] = "/data:ro"
        else:
            ligand_string = ligand

        # Build base command
        cmd = [
            "--protein_path",
            protein_string,
            "--ligand_description",
            ligand_string,
            "--complex_name",
            out_dir_leaf,
            "--out_dir",
            "/results",
            "--samples_per_complex",
            str(num_poses),
        ]

        # Compose kwargs into command arguments
        # Convert kwargs to command-line format: {"key": "value"} -> ["--key", "value"]
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        if verbose:
            logger.info("Mounting volumes:")
            for host_path, container_spec in to_be_mounted_volumes.items():
                logger.info(f"  {host_path} -> {container_spec}")
            logger.info(
                f"Running DiffDock with ligand {ligand} and protein {protein} for complex {out_dir_leaf}"
            )

        try:
            result = manager.run_command(cmd=cmd, volumes=to_be_mounted_volumes)

        except Exception as e:
            logger.error(f"Docker run command failed: {e}")
            raise RuntimeError(f"Docker run command failed. Error: {e}") from e

        return result

    def _run_huggingface(
        self,
        protein: Union[str, Path],
        ligand: Union[str, Path],
        out_dir: Union[str, Path],
        num_poses: int,
        verbose: bool,
        **kwargs,
    ) -> List[PoseResult]:
        """
        Run DiffDock via HuggingFace Spaces.

        Uses the HFSpacesClient (which extends HFClient) for authenticated
        access to the Space and its prediction API.

        Parameters
        ----------
        protein : Union[str, Path]
            Path to PDB file or PDB identifier
        ligand : Union[str, Path]
            SMILES string or path to ligand file
        out_dir : Union[str, Path]
            Directory to save results (may be used by some Spaces)
        num_poses : int
            Number of poses to generate
        verbose : bool, default=True
            Whether to print verbose output
        **kwargs
            Additional parameters passed to HFSpacesClient.predict().
            These can override default parameters or add new ones.
        """

        # Prepare inputs
        protein_input = self._prepare_hf_protein_input(protein)
        ligand_input = self._prepare_hf_ligand_input(ligand)

        out_dir_parent = out_dir.parent
        out_dir_leaf = out_dir.name

        if verbose:
            logger.info(
                f"Running DiffDock via HuggingFace Spaces with ligand {ligand} and protein {protein} for complex {out_dir_leaf}"
            )

        # Call Space API using inherited predict method
        try:
            result = self._hf_client.predict(
                protein_path=protein_input,
                ligand_description=ligand_input,
                complex_name=out_dir_leaf,
                out_dir=out_dir_parent,
                num_poses=num_poses,
                api_name=DIFFDOCK_CONSTANTS.API_NAME,
                **kwargs,  # Pass all kwargs directly to predict method
            )

            return result

        except Exception as e:
            logger.error(f"HuggingFace Spaces prediction failed: {e}")
            raise RuntimeError(
                f"DiffDock prediction failed. This could be due to:\n"
                f"  - Invalid input formats\n"
                f"  - HuggingFace Spaces service issues\n"
                f"  - Network connectivity problems\n"
                f"Error: {e}"
            ) from e

    def _prepare_hf_protein_input(self, protein: Union[str, Path]) -> str:
        """Prepare protein input (PDB file path or sequence)."""
        if isinstance(protein, Path):
            if protein.is_file():
                return str(protein.absolute())
            else:
                raise FileNotFoundError(f"Protein file not found: {protein}")

        else:
            return protein

    def _prepare_hf_ligand_input(self, ligand: Union[str, Path]) -> str:
        """Prepare ligand input (SMILES string or file path)."""

        if isinstance(ligand, Path):
            if ligand.is_file():
                return str(ligand.absolute())
            else:
                raise FileNotFoundError(f"Ligand file not found: {ligand}")
        else:
            return ligand

    def _parse_hf_result(self, result: Any) -> List[PoseResult]:
        """
        Parse HuggingFace Spaces result into PoseResult objects.

        TODO: Implement based on actual Space output format
        """
        return result


class DiffDockManager:
    """
    Manager for DiffDock which handles file loading and caching.

    Attributes
    ----------
    nstructure : nStructure
        The protein structure to dock
    nmol : nMol
        The molecule to dock
    pdb_dir : Path
        The directory where .pdb files are stored
    results_dir : Path
        The directory where diffdock results are stored
    """

    def __init__(
        self, nstructure: nStructure, nmol: nMol, pdb_dir: Path, results_dir: Path
    ):
        self.nstructure = nstructure
        self.nmol = nmol
        self.pdb_dir = pdb_dir
        self.results_dir = results_dir

    @property
    def complex_name(self) -> str:
        return f"{self.nstructure.id}_{self.nmol.smiles}"

    @property
    def pdb_file_path(self) -> Path:
        return self.pdb_dir / self.nstructure.pdb_filename

    def ensure_dock(
        self,
        backend: str = DIFFDOCK_CONSTANTS.BACKENDS.HUGGINGFACE,
        hf_token: Optional[str] = None,
        space_id: Optional[str] = "reginabarzilaygroup/DiffDock-Web",
        diffdock_image: Optional[ImageInfo] = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Use DiffDock to dock the molecule onto the protein structure if results don't exist.

        Parameters
        ----------
        backend : str, default=DIFFDOCK_CONSTANTS.BACKENDS.HUGGINGFACE
            Execution backend:
            - "huggingface": Use HF Spaces API (works on all platforms)
            - "docker": Local Docker image
        hf_token : Optional[str]
            Hugging Face API token
        space_id : Optional[str]
            Hugging Face Space ID for DiffDock
        diffdock_image : Optional[ImageInfo]
            Docker image to use for DiffDock
        verbose : bool, default=True
            Whether to print verbose output
        **kwargs
            Additional parameters passed to _run_docking()

        Returns
        -------
        None
        """

        # ensure that the pdb file is available
        docking_results_dir = self.results_dir / self.complex_name
        if not docking_results_dir.exists():
            self.ensure_pdb_file(verbose)
            self._run_docking(
                backend=backend,
                hf_token=hf_token,
                space_id=space_id,
                diffdock_image=diffdock_image,
                verbose=verbose,
                **kwargs,
            )
        return None

    def ensure_pdb_file(self, verbose: bool = True) -> None:
        """
        Ensure the structure is available as a pdb file in the expected location.

        Parameters
        ----------
        verbose : bool, default=True
            Whether to print verbose output

        Returns
        -------
        None
        """

        pdb_file_path = self.pdb_file_path
        if not pdb_file_path.is_file():
            if verbose:
                logger.info(
                    f"PDB file not found at {pdb_file_path}. Creating it from structure {self.nstructure.id}"
                )
            self.nstructure.to_pdb_file(pdb_file_path)
        return None

    def _run_docking(
        self,
        num_poses: int = 10,
        backend: str = DIFFDOCK_CONSTANTS.BACKENDS.HUGGINGFACE,
        hf_token: Optional[str] = None,
        space_id: Optional[str] = "reginabarzilaygroup/DiffDock-Web",
        diffdock_image: Optional[ImageInfo] = None,
        **kwargs,
    ) -> None:
        """
        Dock the molecule onto the protein structure.

        Parameters
        ----------
        num_poses : int, default=10
            Number of poses to generate
        backend : str, default=DIFFDOCK_CONSTANTS.BACKENDS.HUGGINGFACE
            Execution backend:
            - "huggingface": Use HF Spaces API (works on all platforms)
            - "docker": Local Docker image
        hf_token : Optional[str]
            Hugging Face API token
        space_id : Optional[str]
            Hugging Face Space ID for DiffDock

        Returns
        -------
        None
        """

        pdb_file_path = self.pdb_file_path
        if not pdb_file_path.is_file():
            raise ValueError(
                f"PDB file not found at {pdb_file_path}. Use ensure_pdb_file() to create it."
            )

        DiffDockRunner(
            backend=backend,
            hf_token=hf_token,
            space_id=space_id,
            diffdock_image=diffdock_image,
        ).predict_poses(
            protein=pdb_file_path,
            ligand=self.nmol.smiles,
            out_dir=self.results_dir / self.complex_name,
            num_poses=num_poses,
            **kwargs,
        )


DIFFDOCK_IMAGE = ImageInfo(
    name="diffdock", tag="cpu", registry="local", platform="linux/arm64"
)
