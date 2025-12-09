import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from poke.poke_core import Rayfront
from poke.materials import create_index_model
import poke.plotting as plot

from glasses import sellmeier, S_LAH60, S_BAH27

# Setup files
path_to_files = Path.home() / "Desktop/kmirror-polabs/raytrace_files"
path_to_file = path_to_files / "SCExAO_90.zmx"

# Rayfront parameters
number_of_rays = 20
wavelength = 0.6 # microns
pupil_radius = 8200 / 2 # mm
max_field_of_view = 1 # just to keep un-normalized

aloc = np.array([0.002506290, 0.9999968592, 0.0000000000])
aloc /= np.linalg.norm(aloc)

alocp = np.array([0.0025045662, 0.999308986, 0.0370847260])
alocp /= np.linalg.norm(alocp)

exit_x = np.cross(alocp, aloc)
exit_x /= np.linalg.norm(exit_x)

rf = Rayfront(
        nrays=number_of_rays,
        wavelength=wavelength,
        pupil_radius=pupil_radius,
        max_fov=max_field_of_view,
        circle=False
)

xx = np.linspace(-rf.raybundle_extent, rf.raybundle_extent, number_of_rays)
xx, yy = np.meshgrid(xx, xx)
xx = np.ravel(xx) / pupil_radius
yy = np.ravel(yy) / pupil_radius

# Define some materials from sellmeier coefficients for the ADC
index_slah60 = sellmeier(wavelength, S_LAH60["A"], S_LAH60["B"])
index_sbah27 = sellmeier(wavelength, S_LAH60["A"], S_LAH60["B"])
index_ag = create_index_model("Ag")
index_SiN = create_index_model("SiN")

# Define our coatings
mirror_surface = [(index_SiN(wavelength), 10e-3), (index_ag(wavelength))]

adc_front = (1., index_slah60)
adc_front_rev = (index_slah60, 1.)

adc_back = (1., index_sbah27)
adc_back_rev = (index_sbah27, 1.)

bsc_front = (1., index_SiN(wavelength))
bsc_front_rev = (index_SiN(wavelength), 1.)

# Construct the surface dictionaries
m1 = {"surf": 2, "coating": mirror_surface, "mode": "reflect"}
m2 = {"surf": 3, "coating": mirror_surface, "mode": "reflect"}
m3 = {"surf": 5, "coating": mirror_surface, "mode": "reflect"}

imr1 = {"surf": 11, "coating": mirror_surface, "mode": "reflect"}
imr2 = {"surf": 14, "coating": mirror_surface, "mode": "reflect"}
imr3 = {"surf": 17, "coating": mirror_surface, "mode": "reflect"}

oap1 = {"surf": 21, "coating": mirror_surface, "mode": "reflect"}
fold1 = {"surf": 24, "coating": mirror_surface, "mode": "reflect"}

adc1 = {"surf": 29, "coating": adc_front, "mode": "transmit"}
adc2 = {"surf": 31, "coating": adc_front_rev, "mode": "transmit"}
adc3 = {"surf": 32, "coating": adc_back, "mode": "transmit"}
adc4 = {"surf": 34, "coating": adc_back_rev, "mode": "transmit"}

adc5 = {"surf": 39, "coating": adc_back, "mode": "transmit"}
adc6 = {"surf": 41, "coating": adc_back_rev, "mode": "transmit"}
adc7 = {"surf": 42, "coating": adc_front, "mode": "transmit"}
adc8 = {"surf": 44, "coating": adc_front_rev, "mode": "transmit"}

dm = {"surf": 27, "coating": mirror_surface, "mode": "reflect"}

fold2 = {"surf": 30, "coating": mirror_surface, "mode": "reflect"}
oap2 = {"surf": 33, "coating": mirror_surface, "mode": "reflect"}

bsc1 = {"surf": 59, "coating": bsc_front, "mode": "transmit"}
bsc2 = {"surf": 60, "coating": bsc_front_rev, "mode": "transmit"}

surflist = [
    m1, m2, m3,
    imr1, imr2, imr3,
    oap1, fold1,
    #adc1, adc2, adc3, adc4,
    #adc5, adc6, adc7, adc8,
    dm, fold2, oap2,
    #bsc1, bsc2
]

x0, y0 = rf.base_rays[0], rf.base_rays[1]

rf.as_polarized(surflist)
rf.trace_rayset(str(path_to_file))
rf.compute_jones_pupil(aloc=-aloc, exit_x=exit_x)
plot.jones_pupil(rf)
