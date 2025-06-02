# scfitpy
for "Efficient spectrum analysis for multi-junction nonlinear superconducting circuit"
https://doi.org/10.48550/arXiv.2503.10202

## What's this?

![Image of peak trace and flow of the peak trace program.](./images/image.png)
The extraction of transition frequencies from a spectrum has conventionally relied on empirical methods, and particularly in complex systems it is a time-consuming and cumbersome process. To address this challenge, we establish an semi-automated efficient and precise spectrum analysis method. It, at first, employs image processing methods to extract transition frequencies, subsequently estimates Hamiltonians of superconducting quantum circuit containing multiple Josephson junctions. Additionally, we determine the suitable range of approximations in simulation methods, evaluating the physical reliability of analyses.

## Usage

### Installation

- via github
  1. `pip install "scfitpy[cupy] @ git+https://github.com/AkiyoshiTomonaga/scfitpy.git"`
- local install
  1. `git clone <this repo>`
  1. `pip install .[cupy]`
     - If you don't have a cuda environment, use `pip install .` instead. It doesn't install `cupy`.
- for development
  1. `git clone <this repo>`
  1. `uv sync --dev --extra cupy`


### Advanced Usages

- `./notebooks/001_PeakTrace.ipynb`
- `./notebooks/101_RabiFit.ipynb`
- `./notebooks/102_4JJ_CircFit.ipynb`: TBD
- `./notebooks/201_RabiSpaceCheck.ipynb`: TBD
- `./notebooks/202_QspaceCheck.ipynb`: TBD
- `./notebooks/assets/*`: experimental data to be processed
- `notebooks/outs/*`: processed data

See the [./notebooks](notebooks) dir.
Each notebook corresponds to each step in the paper.


## Acknowledgements

- https://doi.org/10.48550/arXiv.2503.10202

This work was supported by the Japan Society for the Promotion of Science (JSPS) KAKENHI (Grant Numbers JP22K21294 and JP23K13048).

## LICENSE

See the <LICENSE> file.

## Author information

The codes in this repository are written by [Akiyoshi Tomonaga](https://github.com/AkiyoshiTomonaga) and [Kosuke Mizuno](https://github.com/KosukeMizunoAIST).
