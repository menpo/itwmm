# In The Wild 3D Morphable Models

The repository provides the source code of the algorithm of 3D reconstruction of "In-the-Wild" faces in **images** and **videos**, as outlined in the following papers:

> [**3D Reconstruction of "In-the-Wild" Faces in Images and Videos** ---
> J. Booth, A. Roussos, E. Ververas, E. Antonakos, S. Ploumpis, Y. 
Panagagakis, S. Zafeiriou. 
> Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), accepted for publication (2018).](https://doi.org/10.1109/TPAMI.2018.2832138)

> [**3D Face Morphable Models “In-the-Wild”** --- J. Booth, E. Antonakos, S. 
Ploumpis, G. Trigeorgis, Y. Panagagakis, S. Zafeiriou.
CVPR 2017.](https://ibug.doc.ic.ac.uk/media/uploads/documents/booth2017itw3dmm.pdf)

If you use this code, **please cite the above papers**.

The following topics are covered, each one with its own dedicated notebook:

1. Building an "in-the-wild" texture model
2. Creating an expressive 3DMM
3. Fitting "in-the-wild" images
4. Fitting "in-the-wild" videos

### Release Notes (28/06/2018)

- There has been a major update in the code and the problems of the previous preliminary version have been addressed. The current version is fully functional.

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

### Notes 

Please note that this public version of our code has some differences compared to what is described in our aforementioned papers (TPAMI and CVPR):

- The default parameters provided here work in generally quite well, but in our papers these had been fine-tuned in different sets of experiments. 
- The "in-the-wild" texture model that can be extracted following Notebook 1 is a simplified version of the texture model used in our papers. This is because Notebook 1 uses a smaller set of images coming only from (Zhu, Xiangyu, et al. CVPR 2016). Also, the video fitting in Notebook 4 is still using this texture model, instead of the video-specific texture model described in our TPAMI paper. 
- The video fitting in Notebook 4 uses a simpler initialisation than the one described in our TPAMI paper and corresponding supplementary material: this initialisation comes from a per-frame 3D pose estimation using the mean shape of the adopted parametric facial shape model.

These differences result to a simplified version of our image and video fitting algorithms. In practice, we have observed that these simplifications do not have a drastic effect on the accuracy of the results and result to acceptable results. 

### Installation

1. Follow the instructions to [install the Menpo Project with conda](http://www.menpo.org/installation/conda.html).
2. Whilst in the conda environment containing menpo, run `pip install git+https://github.com/menpo/itwmm`.
3. Download a [copy of the code](https://github.com/menpo/itwmm/archive/master.zip) into your Downloads folder.
4. Run `jupyter notebook` and navigate to the `notebooks` directory in the 
downloaded folder.
5. Explore the notebooks in order to understand how to use this codebase.


# Evaluation Code 

We are also providing evaluation code under the folder "evaluation". This will help you evaluate and compare 3D face reconstruction methods using our 3dMDLab & 4DMaja benchmarks (available in our [iBug website](https://ibug.doc.ic.ac.uk/resources/itwmm/)) or other similar benchmarks. 

Please see the demo Notebook evaluation/demo.ipynb ("Reconstruction Evaluation Demo"), where you can also find detailed comments.

