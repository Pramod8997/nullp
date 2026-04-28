"""
REDD data loader stub.

Phase 1 uses synthetic data from scripts/generate_mock_ukdale.py.

Phase 2 implementation:
  - Download REDD from http://redd.csail.mit.edu/
  - Load with nilmtk: DataSet('redd.h5')
  - Extract 1Hz active power for houses 1-6
  - Run through same preprocessing pipeline as synthetic data
  - Segment into window_size=60 windows per appliance
  - Store in same HDF5 structure as mock data

Requires: nilmtk, nilm_metadata (pip install nilmtk)
"""
from typing import Iterator, Tuple


class REDDLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def stream_data(self) -> Iterator[Tuple[float, float]]:
        """Yields (timestamp, power) tuples from REDD HDF5 dataset."""
        raise NotImplementedError(
            "REDD loading is Phase 2 only. "
            "Use scripts/generate_mock_ukdale.py for Phase 1 synthetic data."
        )


def load_redd(path: str):
    """Load REDD dataset. Phase 2 only."""
    raise NotImplementedError("Phase 2 only — see docstring for implementation plan.")
