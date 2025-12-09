# kmirror-polabs
Brief investigation into the retardance incurred by K-mirrors as they rotate

# Assumptions
The question we are trying to answer is _if_ a rotating K-mirror can generate generally elliptical retardation because of cascaded optical elements. My hypothesis is that it does. This model will not be sufficient to quantify the polarization aberrations in SCExAO generally because I am using a rather old raytrace file, before the AO3k and VAMPIRES upgrades. I'm also going to assume that every surface has the same coating for simplicity, the answer to the question will not depend on what coating I use - only the absolute magnitude of the effect.

The base file we are starting with is

```bash
scipath_040827_adc_SCExAO.zmx
```

Which has the pupil derotator at a nominal 90 degrees. I deleted the obstructing apertures, and proceeded with generating a file for each angle from 0 -> 180 deg at 5 deg steps. 
