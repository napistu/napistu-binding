from io import StringIO

import requests
import torch
from Bio.PDB import PDBIO, PDBParser, Structure

from napistu_binding.utils.optional import require_esm


class nStructure(Structure.Structure):
    """Minimal protein structure class extending BioPython Structure.

    Start simple - add methods as needed.
    """

    def __init__(self, structure_id):
        super().__init__(structure_id)

    @classmethod
    def from_pdb_file(cls, pdb_path, structure_id=None):
        """Load from local PDB file.

        Parameters
        ----------
        pdb_path : str
            Path to PDB file
        structure_id : str, optional
            Identifier for structure. Defaults to filename.

        Returns
        -------
        nStructure
        """
        if structure_id is None:
            structure_id = pdb_path.split("/")[-1].replace(".pdb", "")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(structure_id, pdb_path)

        # Create nStructure and copy data
        n = cls(structure.id)
        for model in structure:
            n.add(model)

        return n

    @classmethod
    def from_alphafold(cls, uniprot_id):
        """Download structure from AlphaFold Database.

        Parameters
        ----------
        uniprot_id : str
            UniProt accession (e.g., 'P00338')

        Returns
        -------
        nStructure
        """
        # Use AlphaFold API to get the actual URL
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        response = requests.get(api_url)
        response.raise_for_status()

        data = response.json()

        # Check if structure exists
        if not data:
            raise ValueError(f"No AlphaFold structure found for {uniprot_id}")

        # Get PDB file URL from API response
        pdb_url = data[0]["pdbUrl"]

        # Download PDB file
        pdb_response = requests.get(pdb_url)
        pdb_response.raise_for_status()

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(uniprot_id, StringIO(pdb_response.text))

        n = cls(uniprot_id)
        for model in structure:
            n.add(model)

        return n

    def to_pdb_file(self, output_path):
        """Save structure to PDB file.

        Parameters
        ----------
        output_path : str
            Where to save PDB file
        """
        io = PDBIO()
        io.set_structure(self)
        io.save(output_path)

    @require_esm
    def _compute_esm_embeddings(self, device=None):
        """Compute ESM-2 embeddings for this structure's sequence.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to run computation on. If None, auto-selects.

        Returns
        -------
        torch.Tensor
            Per-residue embeddings [num_residues, 1280]
        """
        import esm

        # Determine device
        if device is None:
            device = self._select_device(mps_valid=True)
        elif isinstance(device, str):
            device = torch.device(device)

        # Load model (cached after first call)
        if self._esm_model is None:
            self._esm_model, self._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self._esm_model.eval()

        # Move model to device
        self._esm_model = self._esm_model.to(device)

        # Prepare sequence
        sequence = self.get_sequence()
        batch_converter = self._alphabet.get_batch_converter()
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        batch_tokens = batch_tokens.to(device)

        REPR_LAYER = 33  # Final layer for esm2_t33_650M_UR50D

        # Get embeddings from final layer
        with torch.no_grad():
            results = self._esm_model(batch_tokens, repr_layers=[REPR_LAYER])

        # Verify we got the expected layer
        assert (
            REPR_LAYER in results["representations"]
        ), f"Expected layer {REPR_LAYER} not in results. Available: {list(results['representations'].keys())}"

        # Extract per-residue embeddings
        # [0]: first sequence in batch
        # [1:-1]: remove BOS/EOS tokens
        # [:]: all embedding dimensions
        embeddings = results["representations"][REPR_LAYER][0, 1:-1, :]

        return embeddings.cpu()
