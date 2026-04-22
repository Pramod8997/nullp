import pandas as pd
from typing import Iterator, Tuple

class SynDLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def stream_data(self) -> Iterator[Tuple[float, float]]:
        """Yields (timestamp, power) tuples."""
        yield 1.0, 10.0
