from dataclasses import dataclass

@dataclass
class Point():
    x: float
    y: float
    z: float = 0.0
    h: int = 1
    w: int = 1

    def __post_init__(self):
        self.x : float =  self.x * self.w
        self.y : float =  self.y * self.h
