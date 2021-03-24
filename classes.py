from dataclasses import dataclass

from typing import Tuple

Layer = Tuple[int]


@dataclass
class NeuronId:
    layer: Tuple[int]
    unit: int

    def url(self) -> str:
        return f"https://microscope.openai.com/models/contrastive_rn50/image_block_{self.layer[0]}_{self.layer[1]}_add_{self.layer[2]}_0/{self.unit}"
