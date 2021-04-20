# Long-term-Motion-in-3D-Scenes

This is an implementation of the CVPR'21 paper "Synthesizing Long-Term 3D Human Motion and Interaction in 3D".

Please check our [paper](https://arxiv.org/pdf/2012.05522.pdf) and the [project webpage](https://jiashunwang.github.io/Long-term-Motion-in-3D-Scenes/) for more details.

## Citation

If you use our code or paper, please consider citing:
```
@article{wang2020synthesizing,
  title={Synthesizing Long-Term 3D Human Motion and Interaction in 3D Scenes},
  author={Wang, Jiashun and Xu, Huazhe and Xu, Jingwei and Liu, Sifei and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2012.05522},
  year={2020}
}
```

## Dependencies

Requirements:
- python3.6
- pytorch==1.1.0
- trimesh
- open3d
- [Chamfer Pytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/tree/719b0f1ca5ba370616cb837c03ab88d9a88173ff)
- [Human Body Prior](https://github.com/nghorbani/human_body_prior)
- [SMPL-X](https://github.com/vchoutas/smplify-x)

## Datasets
We use [PROX](https://prox.is.tue.mpg.de/) and [PROXE](https://github.com/yz-cnsdqz/PSI-release) datasets as our training data. After downloading them, please put them in './data/'. We provide `generate_routepose_data.ipynb` and `generate_sub_data.ipynb` for data generation. Note in PROX, the human meshes and the scene meshes are not in the same area in the world coordinates. Different from PROX and PROXE, we apply the inverse of the camera extrinsics to the scene mesh. Since the scene is the input and we need it to be aligned with the human bodies. This is done in the data generation code. Thus for contact calculating, you do not need to apply transformation to them. While for collision calculating, you still need to apply the transformation to the human bodies similar to [PROXE](https://github.com/yz-cnsdqz/PSI-release) to make it be aligned with SDF. Please be careful with this during training or testing, especially if you want to test on other scenes such as [Matterport3D](https://github.com/niessner/Matterport). Please put body_segments data in './data/' as well.

## Demo
We provide `demo.ipynb` to help you play with 




method. Before running, please put a downsampled `MPH16.ply` mesh and the SDF data of this scene in './demo_data/'. You can download them from [PROX](https://prox.is.tue.mpg.de/) and [PROXE](https://github.com/yz-cnsdqz/PSI-release). Still, please be careful with the camera extrinsics when you want to test other scenes, make sure the human body is in the scene. It will also show you how to optimize the whole motion using our code.

## Models
We use [SMPL-X](https://github.com/vchoutas/smplify-x) to represent human bodies. Please download the SMPL-X models and put them in './models/' and it may look like './models/smplx/SMPLX_NEUTRAL.npz'. Please download [vposer](https://github.com/nghorbani/human_body_prior) model and put it in './'

We also provide our pretrained model [here](https://drive.google.com/file/d/1xRb56tUrrefnffgisWudH8gmKaACX9Ap/view?usp=sharing)

## Training
After you generate the data. You can train the networks directly,
```
python train_subgoal.py
```
```
python train_route.py
```
Please train the posenet after you finished training routenet with your own pretrained routenet model,
```
python train_pose.py
``` 



## Acknowledgement
This work was supported, in part, by grants from DARPA LwLL, NSF 1730158 CI-New: Cognitive Hardware and Software Ecosystem Community Infrastructure (CHASE-CI), NSF ACI-1541349 CC*DNI Pacific Research Platform, and gifts from Qualcomm and TuSimple.
Part of our code is based on [PROXE](https://github.com/yz-cnsdqz/PSI-release) and it may help you with the dependencies and dataset parts as well. Many thanks!

## License
Apache-2.0 License
