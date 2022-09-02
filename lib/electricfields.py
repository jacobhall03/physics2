from vector import Vector
import matplotlib.pyplot as plt
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable
from constants import coulomb_const, elementary_charge, proton_mass, electron_mass

ElectricFieldFunction = Callable[[Vector], Vector]


@dataclass
class PointCharge:
    """A class representing a point charge with position, charge, velocity, and mass attributes."""
    charge: float = elementary_charge
    position: Vector = Vector()
    velocity: Vector = Vector()
    mass: float = proton_mass


class ElectricDipole(tuple):
    """A electric dipole class derived from a tuple
    given a charge, distance apart, and direction vector."""
    def __new__(cls, charge: float = elementary_charge, dist_apart: float = 1.0,
                center_point: Vector = Vector(), axis_direction: Vector = Vector(1)):
        axis_unitvec = axis_direction.unit()
        charge1position = center_point + (axis_unitvec*dist_apart)
        charge2position = center_point - (axis_unitvec*dist_apart)
        charge1 = PointCharge(charge, charge1position)
        charge2 = PointCharge(-charge, charge2position)
        return super().__new__(ElectricDipole, (charge1, charge2))

    def field2torque(self: tuple[PointCharge], electric_field: ElectricFieldFunction) -> Vector:
        """Given an electric field function, returns the torque vector on the dipole
        caused by the electric field."""
        f_c1 = self[0].charge * electric_field(self[0].position)    # Force on charge 1 due to electric field
        r_a1 = (self[0].position - self[1].position) / 2              # Position Vector from axis center to charge 1
        f_c2 = self[1].charge * electric_field(self[1].position)
        r_a2 = -r_a1
        t_1 = r_a1.cross(f_c1)
        t_2 = r_a2.cross(f_c2)
        t_tot = t_1 + t_2
        return t_tot

    def uniformfield2torque(self: tuple[PointCharge], uniform_field: Vector) -> Vector:
        """Given a uniform electric field represented as a vector, returns the torque vector on the dipole
        caused by the electric field."""
        dipole_moment = self[0].charge * (self[0].position - self[1].position)
        torque = dipole_moment.cross(uniform_field)
        return torque

    def uniformfield2energy(self: tuple[PointCharge], uniform_field: Vector) -> float:
        """Given an electric field, returns the potential energy."""
        negdipole_moment = -self[0].charge * (self[0].position - self[1].position)
        energy = negdipole_moment.cross(uniform_field)
        return energy

class ChargeRing():
    """Class representing a charge ring with uniform charge density, radius, and center."""
    def __init__(self, radius: float, charge_density: float, center: Vector = Vector()):
        self.radius = radius
        self.charge_density = charge_density
        self.center = center
    
    def efieldactual(self, location: Vector) -> Vector:
        charge = self.charge_density * 2 * math.pi * self.radius
        disp_vec = (location - self.center)
        eactual_mag = (coulomb_const * charge * abs(disp_vec)) / ((abs(disp_vec))**2 + self.radius**2)**(3.0/2)
        unit_vec = disp_vec / abs(disp_vec)
        return eactual_mag*unit_vec
    
    def efieldsim(self, location: Vector, numcharges: int) -> Vector:
        charge = self.charge_density * 2 * math.pi * self.radius
        chargeperpoint = charge / numcharges
        chargecircle = [PointCharge(chargeperpoint, Vector(self.radius*math.cos(angle), self.radius*math.sin(angle))) for angle in np.linspace(0, 2*math.pi, numcharges)]
        efieldfunc = electricfieldclosure(*chargecircle)
        return efieldfunc(location)


def charges2force(charge1: PointCharge, charge2: PointCharge) -> Vector:
    """Returns the force vector caused by charge 2 on charge1 using coulomb's law"""
    r21 = charge1.position - charge2.position
    try:
        result_mag = coulomb_const * \
            (charge1.charge * charge2.charge) / (abs(r21))**2
        return result_mag * r21.unit()
    except ZeroDivisionError:
        return Vector()


def charge2field(pointcharge: PointCharge, location: Vector) -> Vector:
    """Returns the electric field vector caused by a pointcharge at specified location."""
    r10 = location - pointcharge.position
    try:
        result_mag = (coulomb_const * pointcharge.charge) / (abs(r10))**2
        return result_mag * r10.unit()
    except ZeroDivisionError:
        return Vector()


def electricfieldclosure(*args: PointCharge) -> ElectricFieldFunction:
    """Returns a function representing a electric field caused by point charge(s)."""
    def electricfield(location: Vector) -> Vector:
        resultant = Vector()
        for charge in args:
            resultant += charge2field(charge, location)
        return resultant
    return electricfield

def uniformelectricfieldclosure(efield_vector: Vector) -> ElectricFieldFunction:
    """Returns a function representing a uniform electric field."""
    def uniformelectricfield(location: Vector) -> Vector:
        return efield_vector
    return uniformelectricfield

class ElectricFieldPlotter:
    def __init__(self, *args: PointCharge):
        """Provides a set of methods for analyzing/visualizing a electric field caused by
        given static point charges."""
        self.chargelist = args

    def __mesh2field(self, x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        """Takes a mesh grid formed by x and y parameters, returns ..."""
        u = np.zeros_like(x)
        v = np.zeros_like(y)
        for charge in self.chargelist:
            xdisp = x - charge.position[0]
            ydisp = y - charge.position[1]
            dist = (xdisp**2 + ydisp**2)**(0.5)
            u += (coulomb_const * charge.charge * xdisp) / dist**3
            v += (coulomb_const * charge.charge * ydisp) / dist**3
        return [u, v]

    def plot2Delectricfield(self, dim: list[float] = [-10, 10], num: int = 20) -> None:
        """Plots the unit vectors of the electric field."""
        X, Y = np.meshgrid(np.linspace(*dim, num), np.linspace(*dim, num))
        [u, v] = self.__mesh2field(X, Y)
        mag = (u**2 + v**2)**(0.5)
        u_unit = u / mag
        v_unit = v / mag
        plt.quiver(X, Y, u_unit, v_unit, color='r')
        for charge in self.chargelist:
            plt.plot(charge.position[0], charge.position[1], '-o{color}'.format(color=('r' if charge.charge >= 0.0 else 'k')))
        plt.grid()
        plt.title("Electric Field Unit Vectors")
        plt.show()

    def plot2Dfieldlines(self, dim: list[float] = [-10, 10], num: int = 20) -> None:
        X, Y = np.meshgrid(np.linspace(*dim, num), np.linspace(*dim, num))
        [u, v] = self.__mesh2field(X, Y)
        for charge in self.chargelist:
            plt.plot(charge.position[0], charge.position[1], '-o{color}'.format(color=('r' if charge.charge >= 0.0 else 'k')))
        plt.streamplot(X, Y, u, v, density=1.4, linewidth=None, color='r')
        plt.grid()
        plt.title("Electric Field Lines")
        plt.show()