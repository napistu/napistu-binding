"""
DiffDock molecular docking runner with flexible backend support.

Classes
-------
PoseResult
    Container for a predicted binding pose.
DiffDockRunner
    DiffDock molecular docking runner with flexible backend support.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from napistu_torch.ml.hugging_face import HFSpacesClient

from napistu_binding.constants import DIFFDOCK_CONSTANTS

logger = logging.getLogger(__name__)


class PoseResult:
    """
    Container for a predicted binding pose.

    Attributes
    ----------
    ligand_coords : np.ndarray
        3D coordinates of ligand atoms, shape (n_atoms, 3)
    confidence_score : float
        Confidence score for this pose (higher is better)
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
        - "docker": Local Docker with NVIDIA GPU required (not implemented)
    hf_token : Optional[str]
        HuggingFace API token for authenticated Spaces or rate limit increases
    space_id : Optional[str]
        HuggingFace Space ID for DiffDock. If None, uses official DiffDock Space.

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
        backend: str = DIFFDOCK_CONSTANTS.BACKENDS.HUGGINFACE,
        hf_token: Optional[str] = None,
        space_id: Optional[str] = DIFFDOCK_CONSTANTS.SPACE_ID,
    ):
        """Initialize DiffDock runner with specified backend."""
        self.backend = backend
        self.hf_token = hf_token
        self.space_id = space_id

        # Validate backend choice
        valid_backends = [
            DIFFDOCK_CONSTANTS.BACKENDS.HUGGINFACE,
            DIFFDOCK_CONSTANTS.BACKENDS.DOCKER,
        ]
        if backend not in valid_backends:
            raise ValueError(
                f"Invalid backend: {backend}. " f"Choose from: {valid_backends}"
            )
        elif backend == DIFFDOCK_CONSTANTS.BACKENDS.DOCKER:
            raise NotImplementedError(
                'Docker backend not implemented; use "huggingface" instead.'
            )

        # Initialize HF Spaces client if needed
        # This inherits all the auth validation from HFClient
        self._hf_client = None
        if backend == DIFFDOCK_CONSTANTS.BACKENDS.HUGGINFACE:
            self._hf_client = HFSpacesClient(space_id=self.space_id, hf_token=hf_token)
            logger.info("Initialized DiffDock with HF Spaces backend")

    def predict_poses(
        self,
        protein: Union[str, Path],
        ligand: Union[str, Path],
        num_poses: int = 10,
        **kwargs,
    ) -> List[PoseResult]:
        """
        Predict binding poses for protein-ligand pair.

        Parameters
        ----------
        protein : str or Path
            Path to PDB file or protein sequence string
        ligand : str or Path
            SMILES string or path to ligand file (SDF, MOL2)
        num_poses : int, default=10
            Number of poses to generate
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
        if self.backend == DIFFDOCK_CONSTANTS.BACKENDS.HUGGINFACE:
            return self._run_huggingface(protein, ligand, num_poses, **kwargs)

    def _run_huggingface(
        self,
        protein: Union[str, Path],
        ligand: Union[str, Path],
        num_poses: int,
        **kwargs,
    ) -> List[PoseResult]:
        """
        Run DiffDock via HuggingFace Spaces.

        Uses the HFSpacesClient (which extends HFClient) for authenticated
        access to the Space and its prediction API.
        """
        logger.info("Running DiffDock via HuggingFace Spaces...")

        # Prepare inputs
        protein_input = self._prepare_protein_input(protein)
        ligand_input = self._prepare_ligand_input(ligand)

        # Call Space API using inherited predict method
        try:
            result = self._hf_client.predict(
                protein_path=protein_input,
                ligand_description=ligand_input,
                num_poses=num_poses,
                api_name=DIFFDOCK_CONSTANTS.API_NAME,
            )

            # Parse results
            poses = self._parse_hf_result(result)

            logger.info(f"✓ Generated {len(poses)} poses")
            return poses

        except Exception as e:
            logger.error(f"HuggingFace Spaces prediction failed: {e}")
            raise RuntimeError(
                f"DiffDock prediction failed. This could be due to:\n"
                f"  - Invalid input formats\n"
                f"  - HuggingFace Spaces service issues\n"
                f"  - Network connectivity problems\n"
                f"Error: {e}"
            ) from e

    def _prepare_protein_input(self, protein: Union[str, Path]) -> str:
        """Prepare protein input (PDB file path or sequence)."""
        if isinstance(protein, Path):
            if protein.is_file():
                return str(protein.absolute())
            else:
                raise FileNotFoundError(f"Protein file not found: {protein}")

        else:
            return protein

    def _prepare_ligand_input(self, ligand: Union[str, Path]) -> str:
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
