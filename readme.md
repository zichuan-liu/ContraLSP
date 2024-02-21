# (ICLR 2024) Explaining Time Series via Contrastive and Locally Sparse Perturbations (ContraLSP)

Code for **Explaining Time Series via Contrastive and Locally Sparse Perturbations** accepted by ICLR 2024. ContraLSP is implemented in PyTorch and tested on different time series tasks, including classification and prediction benchmarks. Our experiments are based on [time_interpret](https://github.com/josephenguehard/time_interpret), and thanks to the all original authors! [[paper]](https://openreview.net/pdf?id=qDdSRaOiyb) [[code]](https://github.com/zichuan-liu/ContraLSP)

## Citing ContraLSP
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```tex
@inproceedings{liu2024explaining,
      title={Explaining Time Series via Contrastive and Locally Sparse Perturbations}, 
      author={Zichuan Liu and Yingying Zhang and Tianchun Wang and Zefan Wang and Dongsheng Luo and Mengnan Du and Min Wu and Yi Wang and Chunlin Chen and Lunting Fan and Qingsong Wen},
      year={2024},
      booktitle={Proceedings of the 12th International Conference on Learning Representations},
      pages={1-21}
}
```
In case of any questions, bugs, suggestions or improvements, please feel free to drop me at _zichuanliu@smail.nju.edu.cn_ or open an issue.

 


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







## Further Reading
1, [**Transformers in Time Series: A Survey**](https://arxiv.org/abs/2202.07125), in IJCAI 2023.
[\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)

**Authors**: Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, Liang Sun

```bibtex
@inproceedings{wen2023transformers,
  title={Transformers in time series: A survey},
  author={Wen, Qingsong and Zhou, Tian and Zhang, Chaoli and Chen, Weiqi and Ma, Ziqing and Yan, Junchi and Sun, Liang},
  booktitle={International Joint Conference on Artificial Intelligence(IJCAI)},
  year={2023}
}
```

2, [**Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook**](https://arxiv.org/abs/2310.10196), in *arXiv* 2023.
[\[GitHub Repo\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)

**Authors**: Ming Jin, Qingsong Wen*, Yuxuan Liang, Chaoli Zhang, Siqiao Xue, Xue Wang, James Zhang, Yi Wang, Haifeng Chen, Xiaoli Li (IEEE Fellow), Shirui Pan*, Vincent S. Tseng (IEEE Fellow), Yu Zheng (IEEE Fellow), Lei Chen (IEEE Fellow), Hui Xiong (IEEE Fellow)

```bibtex
@article{jin2023lm4ts,
  title={Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook}, 
  author={Ming Jin and Qingsong Wen and Yuxuan Liang and Chaoli Zhang and Siqiao Xue and Xue Wang and James Zhang and Yi Wang and Haifeng Chen and Xiaoli Li and Shirui Pan and Vincent S. Tseng and Yu Zheng and Lei Chen and Hui Xiong},
  journal={arXiv preprint arXiv:2310.10196},
  year={2023}
}
```

3, [**Position Paper: What Can Large Language Models Tell Us about Time Series Analysis**](https://arxiv.org/abs/2402.02713), in *arXiv* 2024.

**Authors**: Ming Jin, Yifan Zhang, Wei Chen, Kexin Zhang, Yuxuan Liang*, Bin Yang, Jindong Wang, Shirui Pan, Qingsong Wen*


```bibtex
@article{jin2024position,
   title={Position Paper: What Can Large Language Models Tell Us about Time Series Analysis}, 
   author={Ming Jin and Yifan Zhang and Wei Chen and Kexin Zhang and Yuxuan Liang and Bin Yang and Jindong Wang and Shirui Pan and Qingsong Wen},
  journal={arXiv preprint arXiv:2402.02713},
  year={2024}
}
```
4, [**AI for Time Series (AI4TS) Papers, Tutorials, and Surveys**](https://github.com/qingsongedu/awesome-AI-for-time-series-papers)
