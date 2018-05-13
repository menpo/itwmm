# In The Wild 3D Morphable Models

The repository provides an implementation of
In-The-Wild 3D Morphable Models as outlined in our CVPR paper:

> [**3D Face Morphable Models “In-the-Wild”** --- J. Booth, E. Antonakos, S. 
Ploumpis, G. Trigeorgis, Y. Panagagakis, S. Zafeiriou.
CVPR 2017](https://ibug.doc.ic.ac.uk/media/uploads/documents/booth2017itw3dmm.pdf)

and our journal extension:

> **3D Reconstruction of "In-the-Wild" Faces in Images and Videos** ---
> J. Booth, A. Roussos, E. Ververas, E. Antonakos, S. Ploumpis, Y. 
Panagagakis S. Zafeiriou. 
> TPAMI 2018

The following topics are covered, each one with it's own dedicated notebook:

1. Building an "in-the-wild" texture model
2. Creating an expressive 3DMM
3. Fitting "in-the-wild" images
4. Fitting "in-the-wild" videos

### Prerequisites

To leverage this codebase users need to independently source the 
following items to construct an "in-the-wild" 3DMM:

- A collection of "in-the-wild" images coupled with 3D fits
- A parametric facial shape model of identity and expression

And then to use this model, users will need to provide data to fit on:

- "in-the-wild" images or videos with iBUG 68 annotations

Examples are given for working with some common facial models (e.g. LSFM) and 
it shouldn't be too challenging to adapt these examples for alternative inputs.
Just bear in mind that fitting parameters will need to be tuned when working 
with different models.

### Installation

1. Follow the instructions to [install the Menpo Project with conda](http://www.menpo.org/installation/conda.html).
2. Whilst in the conda environment containing menpo, run `pip install git+https://github.com/menpo/itwmm`.
3. Download a [copy of the code](https://github.com/menpo/itwmm/archive/master.zip) into your Downloads folder.
4. Run `jupyter notebook` and navigate to the `notebooks` directory in the 
downloaded folder.
5. Explore the notebooks in order to understand how to use this codebase.
