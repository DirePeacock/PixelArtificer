import colorsys

# colorsys uses floats 0.0f-1.0f and PIL expects ints 0-255
PIL_MAX = 255


class Color:
    """Color object for use with PIL and colorsys"""

    def __init__(self, rgba=(0, 0, 0, 0), hsv=None):
        if hsv is None:
            if isinstance(rgba, tuple):
                self._set_rgba(*rgba)
            else:
                self.hex = rgba
        else:
            self._set_hsv(*hsv)

    @property
    def rgba(self):
        return tuple(self.r, self.g, self.b, self.a)

    def _set_rgba(self, r, g, b, a=PIL_MAX):
        self.r = int(r * PIL_MAX) if isinstance(r, float) else r
        self.g = int(g * PIL_MAX) if isinstance(g, float) else g
        self.b = int(b * PIL_MAX) if isinstance(b, float) else b
        self.a = int(a * PIL_MAX) if isinstance(a, float) else a

    @property
    def colorsys_rgba(self):
        return tuple(
            self.r / PIL_MAX, self.g / PIL_MAX, self.b / PIL_MAX, self.a / PIL_MAX
        )

    @rgba.setter
    def rgba(self, rgba):
        self._set_rgba(*rgba)
        hsv = self.rgb_to_hsv(self.rgba)
        self._set_hsv(hsv)

    @property
    def hsv(self):
        return tuple(self.h, self.s, self.v)

    def _set_hsv(self, h, s, v):
        self.h = h if isinstance(h, float) else h / PIL_MAX
        self.s = s if isinstance(s, float) else s / PIL_MAX
        self.v = v if isinstance(v, float) else v / PIL_MAX

    @hsv.setter
    def hsv(self, hsv):
        self._set_hsv(*hsv)
        rgba = self.hsv_to_rgb(self.hsv)
        self._set_rgba(*rgba)

    def rgb_to_hsv(self, rgb):
        return colorsys.rgb_to_hsv(self.r / PIL_MAX, self.g / PIL_MAX, self.b / PIL_MAX)

    def hsv_to_rgb(self, hsv):
        retval = colorsys.hsv_to_rgb(*hsv)
        retval = tuple(int(x * PIL_MAX) for x in retval)

    @property
    def hex(self):
        new_hex = 0x00000000
        new_hex |= self.r << 24
        new_hex |= self.g << 16
        new_hex |= self.b << 8
        new_hex |= self.a
        return new_hex

    @hex.setter
    def hex(self, hex):
        self._set_hex_rgba(hex)

    def _set_hex_rgba(self, hex):
        _r = (hex >> 24) & 0xFF
        _g = (hex >> 16) & 0xFF
        _b = (hex >> 8) & 0xFF
        _a = hex & 0xFF
        self._set_rgba(_r, _g, _b, _a)

    # operators
    def __repr__(self):
        return f"Color(rgba={self.rgba}, hsv={self.hsv})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.rgba == other.rgba

    def __ne__(self, other):
        return self.rgba != other.rgba

    def __add__(self, other):
        _r = min(PIL_MAX, self.r + other.r)
        _g = min(PIL_MAX, self.g + other.g)
        _b = min(PIL_MAX, self.b + other.b)
        _a = min(PIL_MAX, self.a + other.a)
        return Color(rgba=(_r, _g, _b, _a))

    def __sub__(self, other):
        _r = max(0, self.r - other.r)
        _g = max(0, self.g - other.g)
        _b = max(0, self.b - other.b)
        _a = max(0, self.a - other.a)
        return Color(rgba=(_r, _g, _b, _a))

    def __mul__(self, other):
        _r = min(PIL_MAX, self.r * other)
        _g = min(PIL_MAX, self.g * other)
        _b = min(PIL_MAX, self.b * other)
        _a = min(PIL_MAX, self.a * other)
        return Color(rgba=(_r, _g, _b, _a))

    def __truediv__(self, other):
        _r = max(0, self.r / other)
        _g = max(0, self.g / other)
        _b = max(0, self.b / other)
        _a = max(0, self.a / other)
        return Color(rgba=(_r, _g, _b, _a))

    def __floordiv__(self, other):
        _r = max(0, self.r // other)
        _g = max(0, self.g // other)
        _b = max(0, self.b // other)
        _a = max(0, self.a // other)
        return Color(rgba=(_r, _g, _b, _a))
