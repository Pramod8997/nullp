"""
UK-DALE data loader stub.

Phase 1 uses synthetic data from scripts/generate_mock_ukdale.py.

Phase 2 implementation:
  - Download UK-DALE from https://data.ukedc.rl.ac.uk/
  - Load with nilmtk: DataSet('ukdale.h5')
  - Extract 1Hz active power for houses 1-5
  - Run through same preprocessing pipeline as synthetic data
  - Segment into window_size=60 windows per appliance
  - Store in same HDF5 structure as mock data:
      /appliances/{class_name}/windows  — shape [N, 60]
      /appliances/{class_name}/labels   — integer class index
      /metadata/class_names
      /metadata/sample_rate_hz

Requires: nilmtk, nilm_metadata (pip install nilmtk)
"""
from typing import Iterator, Tuple


class UKDaleLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def stream_data(self) -> Iterator[Tuple[float, float]]:
        """
        Yields (timestamp, power) tuples from UK-DALE HDF5 dataset.

        Phase 2:
          dataset = DataSet(self.data_path)
          elec = dataset.buildings[1].elec
          for meter in elec.meters:
              for chunk in meter.power_series(ac_type='active', sample_period=1):
                  for ts, power in chunk.items():
                      yield (ts.timestamp(), power)
        """
        raise NotImplementedError(
            "UK-DALE loading is Phase 2 only. "
            "Use scripts/generate_mock_ukdale.py for Phase 1 synthetic data."
        )


def load_ukdale(path: str):
    """Load UK-DALE dataset. Phase 2 only."""
    raise NotImplementedError("Phase 2 only — see docstring for implementation plan.")
