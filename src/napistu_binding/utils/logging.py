"""
Utilities for controlling logging behavior.

Public Functions
----------------
suppress_rdkit_logging:
    Disable RDKit warnings, errors, info, and debug messages.
restore_rdkit_logging:
    Re-enable RDKit warnings, errors, info, and debug messages.
"""

from napistu_binding.utils.optional import require_rdkit


@require_rdkit
def restore_rdkit_logging():
    """Re-enable RDKit warnings, errors, info, and debug messages."""
    from rdkit import RDLogger

    RDLogger.EnableLog("rdApp.warning")
    RDLogger.EnableLog("rdApp.error")
    RDLogger.EnableLog("rdApp.info")
    RDLogger.EnableLog("rdApp.debug")


@require_rdkit
def suppress_rdkit_logging():
    """Disable RDKit warnings, errors, info, and debug messages."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.info")
    RDLogger.DisableLog("rdApp.debug")
