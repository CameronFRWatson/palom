# Piecewise alignment for layers of mosaics

Palom started as a tool for registering [whole-slide
images](https://en.wikipedia.org/wiki/Digital_pathology) of the same [FFPE
section](https://en.wikipedia.org/wiki/Histology#Sample_preparation) with
different [IHC stainings](https://en.wikipedia.org/wiki/Immunohistochemistry).

---

## Installation

Installing palom in a fresh conda environment is recommended. [Instruction for
installing miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Step 1

Create a named conda environment - _palom_, in the following example, and activate the environment.

```
conda create -n palom python=3.7 pip -c conda-forge
conda activate palom
```

### Step 2

Install openslide in the conda environment.

```
conda install openslide -c sdvillal
```

### Step 3

Install palom from pypi in the conda environment.

```
python -m pip install palom
```

---

## CLI usage

### Configuration YAML file

Palom CLI tool merges multiple SVS files into a pyramidal OME-TIFF file, with
the option to perform preset stain separation (4 modes are available - `output
mode`: `hematoxylin`, `aec`, `grayscale`, `color`)

A user-defined configuration YAML file is required for the run. A configuration
example can be printed to the console by running

```bash
palom-svs show example
```

```yaml
input dir: Y:\user\me\projects\data\mihc
output full path: Y:\user\me\projects\analysis\mihc\2021\skin-case-356.ome.tif

reference image:
    filename: 20210111/skin_case_356_HEM_C11R3_HEM.svs
    output mode: hematoxylin
    channel name: Hematoxylin

moving images:
- filename: 20210101/skin_case_356_HEM_C01R1_PD1.svs
  output mode: aec
  channel name: PD-1
- filename: 20210101/skin_case_356_HEM_C01R2_PDL1.svs
  output mode: aec
  channel name: PD-L1
```

To show the configuration schema, run the following command

```bash
palom-svs show schema
```

### Use the helper script to generate the configuration file

[A helper
script](https://github.com/Yu-AnChen/palom/blob/main/palom/cli/helper.py) is
included showing how to automatically generate the configuration file if the SVS
files are organized and have specific naming pattern.

Here's an example directory containing many SVS files

```
Y:\DATA\SARDANA\MIHC\768473\RAW
    CBB_SARDANA_768473_C04R1_CD8.svs
    KB_SARDANA_768473_C01R1_PD1.svs
    KB_SARDANA_768473_C01R2_PDL1.svs
    KB_SARDANA_768473_C01R3_Hem.svs
    KB_SARDANA_768473_C02R1_CD4.svs
    KB_SARDANA_768473_C03R1_CD3.svs
    KB_SARDANA_768473_C03R3_DCLAMP.svs
```

Running the following command to generate the configuration file

```bash
palom-svs-helper -i "Y:\DATA\SARDANA\MIHC\768473\RAW" -n "*Hem*" -o "Y:\DATA\SARDANA\MIHC\768473\RAW\palom\768473.ome.tif" -c "Y:\DATA\SARDANA\MIHC\768473\768473.yml"
```

And the resulting `Y:\DATA\SARDANA\MIHC\768473\768473.yml` file 

```yaml
input dir: Y:\DATA\SARDANA\MIHC\768473\RAW
output full path: Y:\DATA\SARDANA\MIHC\768473\RAW\palom\768473.ome.tif
reference image:
  filename: .\KB_SARDANA_768473_C01R3_Hem.svs
  output mode: hematoxylin
  channel name: Hem-C01R3
moving images:
- filename: .\KB_SARDANA_768473_C01R1_PD1.svs
  output mode: aec
  channel name: PD1-C01R1
- filename: .\KB_SARDANA_768473_C01R2_PDL1.svs
  output mode: aec
  channel name: PDL1-C01R2
- filename: .\KB_SARDANA_768473_C02R1_CD4.svs
  output mode: aec
  channel name: CD4-C02R1
- filename: .\KB_SARDANA_768473_C03R1_CD3.svs
  output mode: aec
  channel name: CD3-C03R1
- filename: .\KB_SARDANA_768473_C03R3_DCLAMP.svs
  output mode: aec
  channel name: DCLAMP-C03R3
- filename: .\CBB_SARDANA_768473_C04R1_CD8.svs
  output mode: aec
  channel name: CD8-C04R1
```

After reviewing the configuration file, process those SVS files by running

```bash
palom-svs run -c "Y:\DATA\SARDANA\MIHC\768473\768473.yml"
```

When the process is finished, a pyramidal OME-TIFF file will be generated along
with PNG files showing the feature-based registration results and a log file. 

```
Y:\DATA\SARDANA\MIHC\768473\RAW
│   CBB_SARDANA_768473_C04R1_CD8.svs
│   KB_SARDANA_768473_C01R1_PD1.svs
│   KB_SARDANA_768473_C01R2_PDL1.svs
│   KB_SARDANA_768473_C01R3_Hem.svs
│   KB_SARDANA_768473_C02R1_CD4.svs
│   KB_SARDANA_768473_C03R1_CD3.svs
│   KB_SARDANA_768473_C03R3_DCLAMP.svs
│
└───palom
    │   768473.ome.tif
    │
    └───qc
            01-KB_SARDANA_768473_C01R1_PD1.svs.png
            02-KB_SARDANA_768473_C01R2_PDL1.svs.png
            03-KB_SARDANA_768473_C02R1_CD4.svs.png
            04-KB_SARDANA_768473_C03R1_CD3.svs.png
            05-KB_SARDANA_768473_C03R3_DCLAMP.svs.png
            06-CBB_SARDANA_768473_C04R1_CD8.svs.png
            768473.ome.tif.log
```

---

## Scripting

__WARNING__ API may change in the future

### For SVS files

```python
import palom

c1r = palom.reader.SvsReader(r'Y:\DATA\SARDANA\MIHC\75684\GG_TNP_75684_D21_C11R3_HEM.svs')
c2r = palom.reader.SvsReader(r'Y:\DATA\SARDANA\MIHC\75684\GG_TNP_75684_D23_C01R1_PD1.svs')

c1rp = palom.color.PyramidHaxProcessor(c1r.pyramid, thumbnail_level=2)
c2rp = palom.color.PyramidHaxProcessor(c2r.pyramid, thumbnail_level=2)

LEVEL = 1
c21l = palom.align.Aligner(
    c1rp.get_processed_color(LEVEL), 
    c2rp.get_processed_color(LEVEL),
    ref_thumbnail=c1rp.get_processed_color(2).compute(),
    moving_thumbnail=c2rp.get_processed_color(2).compute(),
    ref_thumbnail_down_factor=c1r.level_downsamples[2] / c1r.level_downsamples[LEVEL],
    moving_thumbnail_down_factor=c2r.level_downsamples[2] / c2r.level_downsamples[LEVEL]
)

c21l.coarse_register_affine()
c21l.compute_shifts()
c21l.constrain_shifts()

c21l.block_affine_matrices_da

c2m = palom.align.block_affine_transformed_moving_img(
    c1rp.get_processed_color(LEVEL),
    c2rp.get_processed_color(LEVEL, 'aec'),
    mxs=c21l.block_affine_matrices_da
)

palom.pyramid.write_pyramid(
    palom.pyramid.normalize_mosaics([c2m]),
    r"Y:\DATA\SARDANA\MIHC\75684\mosaic.ome.tif"
)
```

### For OME-TIFF files

```python
import palom

c1r = palom.reader.OmePyramidReader(r"Z:\P37_Pilot2\P37_S12_Full.ome.tiff")
c2r = palom.reader.OmePyramidReader(r"Z:\P37_Pilot2\HE\P37_S12_E033_93_HE.ome.tiff")

LEVEL = 1
c21l = palom.align.Aligner(
    c1r.read_level_channels(LEVEL, 0),
    c2r.read_level_channels(LEVEL, 1),
    c1r.read_level_channels(3, 0).compute(),
    c2r.read_level_channels(3, 1).compute(),
    c1r.level_downsamples[3] / c1r.level_downsamples[LEVEL],
    c2r.level_downsamples[3] / c2r.level_downsamples[LEVEL]
)

c21l.coarse_register_affine(n_keypoints=4000)
c21l.compute_shifts()
c21l.constrain_shifts()

c2m = palom.align.block_affine_transformed_moving_img(
    c1r.read_level_channels(LEVEL, 0),
    c2r.pyramid[LEVEL],
    mxs=c21l.block_affine_matrices_da
)

palom.pyramid.write_pyramid(
    palom.pyramid.normalize_mosaics([
        c1r.read_level_channels(LEVEL, [0, 1, 2]),
        c2m
    ]),
    r"Z:\P37_Pilot2\mosaic.ome.tif"
)
```