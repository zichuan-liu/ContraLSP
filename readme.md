# (ICLR 2024) Explaining Time Series via Contrastive and Locally Sparse Perturbations (ContraLSP)

Code for **Explaining Time Series via Contrastive and Locally Sparse Perturbations** accepted by ICLR 2024. ContraLSP is implemented in PyTorch and tested on different time series tasks, including classification and prediction benchmarks. Our experiments are based on [time_interpret](https://github.com/josephenguehard/time_interpret), and thanks to the all original authors! [[paper]](https://openreview.net/pdf?id=qDdSRaOiyb) [[code]](https://github.com/zichuan-liu/ContraLSP)


## Installation instructions

```shell script
pip install -r requirement.txt
```
The requirements.txt file can be used to install the necessary packages into a virtual environment.

## Run our toy data

```shell script
cd ContraLSP
python demo.py
```
This is a test explaining a simple white-box regression model, in which you can see a comparison of three different masks (including dynamask, nnmask, and ours).

## Reproducing experiments

All of our experiments can be reproduced by following the instructions.

**Go to the relevant dataset directory (rare/hmm/switchstate/mortalty) and execute python main.py**, e.g.:

run the diffgroup rare feature dataset:
```shell script
cd rare
python rare_feature_diffgroup.py --print_result False
```
and then test it and print results:
```
python rare_feature_diffgroup.py --print_result True
```

All results will be stored in the `save_dir` folder.


## Citing ContraLSP

If you find this repository useful for your research, please cite it in BibTeX format:

```tex
@article{liu2024explaining,
      title={Explaining Time Series via Contrastive and Locally Sparse Perturbations}, 
      author={Zichuan Liu and Yingying Zhang and Tianchun Wang and Zefan Wang and Dongsheng Luo and Mengnan Du and Min Wu and Yi Wang and Chunlin Chen and Lunting Fan and Qingsong Wen},
      year={2023},
      journal={arXiv preprint arXiv:2401.08552}
}
```
In case of any questions, bugs, suggestions or improvements, please feel free to drop me at _zichuanliu@smail.nju.edu.cn_ or open an issue.
