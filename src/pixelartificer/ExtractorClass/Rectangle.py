class Rectangle():
    def __init__(self, x,y,w,h):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
    
    @property
    def key(self):
        return (f"{self.x},{self.y},{self.w},{self.h}")
    def __repr__(self):
        return self.key
    def __str__(self):
        return self.__repr__()
    @classmethod
    def from_key(cls, key):
        x,y,w,h = key.split(",")
        return cls(int(x), int(y), int(w), int(h))