# SparseMPN-for-Multiple-Object-Tracking
This project implement our method "Sparse Message Passing Network with Feature Integration for Online Multiple Object Tracking".

### results
Here is a short example video of the tracking result.

### Environment: 

The basic environment is based on the following link: https://github.com/dvl-tum/mot_neural_solver You need to install related tools to run the code.

### Dataset: 

You need to download the MOTChallenge dataset and prepare the detections boxes in advance, and put the data in corresponding directories (MOT16 and detection). Here we provide a link for downloading the dataset directly. MOTChallenge dataset: https://drive.google.com/file/d/1l-R-2sqSdWgjuXSdRXADWEtiRLS03Nep/view?usp=drive_link Detection boxes (https://drive.google.com/file/d/1q5V-7siYWbR7afaDCmLzVV7J1HEUnh2x/view?usp=drive_link)

### Feature Extraction: 

Running the feat_extract.py to extract the features for each person. To run this code, you need to download two models (https://drive.google.com/drive/folders/1n2u8MokCaAiipktU5Vs4XDuHyq6YcVgX?usp=drive_link) and put them into the directory. This may take a little time. Feature extraction is based on "Bag of tricks and a strong baseline for deep person re-identification" You can also choose to download the features directly from the following link. https://drive.google.com/file/d/1l-R-2sqSdWgjuXSdRXADWEtiRLS03Nep/view?usp=drive_link

### SparseMPN 
Running "train.py" to train our model or running "test.py" to directly do the inference.

The method is proposed based on the work "Learning a Neural Solver for Multiple Object Tracking (CVPR 2020)". Based on this work, we redesigned the graph construct process and rewrite the inference code. Besides, we use a new feature extraction and integration method. Details can be seen from our paper: Sparse Message Passing Network with Feature Integration for Online Multiple Object Tracking.

If you find this work useful, please cite our work: 

@article{wang2022sparse, title={Sparse Message Passing Network with Feature Integration for Online Multiple Object Tracking}, author={Wang, Bisheng and Possegger, Horst and Bischof, Horst and Cao, Guo}, journal={arXiv preprint arXiv:2212.02992}, year={2022} }

### Reference 

@inproceedings{braso2020learning, title={Learning a neural solver for multiple object tracking}, author={Bras{\'o}, Guillem and Leal-Taix{\'e}, Laura}, booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition}, pages={6247--6257}, year={2020} } 

@inproceedings{luo2019bag, title={Bag of tricks and a strong baseline for deep person re-identification}, author={Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei}, booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops}, pages={0--0}, year={2019} }
