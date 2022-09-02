import math

class Vector(tuple):
    '''A 3D Vector class derived from a tuple.

    Provides for vectors, v1 and v2, and a number, k:
        v1 + v2 : Vector Addition
        v1 - v2 : Vector Substraction
        v1*v2 : Dot Product
        k*a, a*k : Scalar Multiplication
        abs(v1) : Absolute Value (Magnitude)
        -v1 \ -v2 : Negative Vector
        v1.rotatexy(anglerad) : Rotation in xy plane (radians)
        v1.rotatexy_deg(angledeg) : Rotation in xy plane (degrees)
        v1.cross(v2) : Cross product of v1 and v2 (v1 X v2)
        v1.unit() : Unit vector with direction of v1
        '''
    def __new__(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        return tuple.__new__(Vector, (x, y, z))    

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self[0] + other[0], self[1] + other[1], self[2] + other[2])
        elif (isinstance(other, list) or isinstance(other, tuple)) and len(other) == 3:
            return Vector(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self[0] - other[0], self[1] - other[1], self[2] - other[2])
        elif (isinstance(other, list) or isinstance(other, tuple)) and len(other) == 3:
            return Vector(self[0] - other[0], self[1] - other[1], self[2] - other[2])

    def __mul__(self, other):
        if isinstance(other, Vector):
            # Dot Product
            return self[0]*other[0] + self[1]*other[1] + self[2]*other[2]
        elif isinstance(other, int) or isinstance(other, float):
            # Scalar Product
            return Vector(self[0]*other, self[1]*other, self[2]*other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            # Scalar Product
            return Vector(self[0]*other, self[1]*other, self[2]*other)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector(self[0] / other, self[1] / other, self[2] / other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector(other / self[0], other / self[1], other / self[2])
        return NotImplemented

    def __abs__(self) -> float:
        return (self[0]**2 + self[1]**2 + self[2]**2)**0.5

    def __neg__(self):
        return Vector(-self[0], -self[1], -self[2])

    def __matmul__(self, other: 'Vector'):
        if (isinstance(other, Vector) or isinstance(other, tuple) or isinstance(other, list)) and len(other) == 3:
            s1 = self[1]*other[2] - self[2]*other[1]
            s2 = self[2]*other[0] - self[0]*other[2]
            s3 = self[0]*other[1] - self[1]*other[0]
            return Vector(s1, s2, s3)

    def rotatexy(self, anglerad: float = 0.0):
        """Rotates self in xy plane, counterclockwise by angle(radians)"""
        newx = self[0]*math.cos(anglerad) - self[1]*math.sin(anglerad)
        newy = self[0]*math.sin(anglerad) + self[1]*math.cos(anglerad)
        return Vector(newx, newy, self[2])

    def rotatexy_deg(self, angledeg: float = 0.0):
        """Rotates self in xy plane, counterclockwise by angle(degrees)"""
        anglerad = angledeg * math.pi / 180.0
        return self.rotatexy(anglerad)

    def cross(self, other: 'Vector'):
        """Calculates cross product with more explicit usage syntax."""
        if (isinstance(other, Vector) or isinstance(other, tuple) or isinstance(other, list)) and len(other) == 3:
            return self.__matmul__(other)

    def unit(self):
        return self * (1.0 / self.__abs__())

    def __repr__(self):
        for r in self:
            if ((abs(r) >= 1.0e3) or (abs(r) <= 1.0e-3)):
                return "(%.3e, %.3e, %.3e)" % self
        return "(%.3f, %.3f, %.3f)" % self


