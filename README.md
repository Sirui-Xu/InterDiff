<p align="center">
<!-- <h1 align="center">InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion</h1> -->
<strong>InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion</strong></h1>
  <p align="center">
    <a href='https://sirui-xu.github.io/' target='_blank'>Sirui Xu</a>&emsp;
    <a href='https://github.com/lzhyu' target='_blank'>Zhengyuan Li</a>&emsp;
    <a href='https://yxw.web.illinois.edu/' target='_blank'>Yu-Xiong Wang*</a>&emsp;
    <a href='https://cs.illinois.edu/about/people/faculty/lgui' target='_blank'>Liang-Yan Gui*</a>&emsp;
    <br>
    University of Illinois Urbana-Champaign
    <br>
    ICCV 2023
  </p>
</p>

</p>
<p align="center">
  <a href='https://arxiv.org/abs/2308.16905'>
    <img src='https://img.shields.io/badge/Arxiv-2308.16905-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/abs/2308.16905.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://sirui-xu.github.io/InterDiff/'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
  <a href='https://youtu.be/Ako1n9HEGBo'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a>
  <a href='https://www.bilibili.com/video/BV1Xh4y187ZH/'>
    <img src='https://img.shields.io/badge/Bilibili-Video-4EABE6?style=flat&logo=Bilibili&logoColor=4EABE6'></a>
  <a href='https://zhuanlan.zhihu.com/p/657497860'>
    <img src='https://img.shields.io/badge/Zhihu-Doc-2F6BE0?style=flat&logo=Zhihu&logoColor=2F6BE0'></a> 
  <a href='https://github.com/Sirui-Xu/InterDiff'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=Sirui-Xu.InterDiff&left_color=gray&right_color=orange">
  </a>
</p>

## 🏠 About
<div style="text-align: center;">
    <img src="assets/teaser.png" width=100% >
</div>
This paper addresses a novel task of anticipating 3D human-object interactions (HOIs). Most existing research on HOI synthesis lacks comprehensive whole-body interactions with dynamic objects, e.g., often limited to manipulating small or static objects. Our task is significantly more challenging, as it requires <b>modeling dynamic objects with various shapes</b>, <b>capturing whole-body motion</b>, and <b>ensuring physically valid interactions</b>. To this end, we propose InterDiff, a framework comprising two key steps: (i) interaction diffusion, where we leverage a diffusion model to encode the distribution of future human-object interactions; (ii) interaction correction, where we introduce a physics-informed predictor to correct denoised HOIs in a diffusion step. Our key insight is to <b>inject prior knowledge</b> that the <b>interactions under reference with respect to contact points</b> follow a simple pattern and are easily predictable. Experiments on multiple human-object interaction datasets demonstrate the effectiveness of our method for this task, capable of producing realistic, vivid, and remarkably long-term 3D HOI predictions.

## 📹 Demo
<p align="center">
    <img src="assets/teaser.gif" align="center" width=60% >
</p>

## 🔥 News
- [2023-09-16] Release a [demo video](https://sirui-xu.github.io/InterDiff/images/teaser_1080p.mp4) 📹.
- [2023-09-01] Our paper is available on the [Arxiv](https://arxiv.org/pdf/2308.16905) &#x1F389; Code/Models are coming soon. Please stay tuned! ☕️

## 📝 TODO List
- [ ] Release more demos.
- [ ] Release training and evaluation code.
- [ ] Release code for visualization.

## 🔍 Overview

### 💡 Key Insight
<p align="center">
  <img src="assets/5.gif" width="50%"/><img src="assets/24.gif" width="50%"/>
  <img src="assets/50.gif" width="50%"/><img src="assets/38.gif" width="50%"/>
</p>
We present HOI sequences (<b>left</b>), object motions (<b>middle</b>), and objects relative to the contacts after coordinate transformations (<b>right</b>). Our key insight is to inject coordinate transformations into a diffusion model, as the relative motion shows simpler patterns that are easier to predict, e.g., being almost stationary (<b>top</b>), or rotating around a fixed axis (<b>bottom</b>).
<!-- 
### 🧪 Experimental Results

#### Qualitative Comparisons with Pure Diffusion
<p align="center">
  <img src="assets/results.png" align="center" width="100%">
</p> -->


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{
   xu2023interdiff,
   title={{InterDiff}: Generating 3D Human-Object Interactions with Physics-Informed Diffusion},
   author={Xu, Sirui and Li, Zhengyuan and Wang, Yu-Xiong and Gui, Liang-Yan},
   booktitle={ICCV},
   year={2023},
}
```


## 👏 Acknowledgements
- [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/): We use the BEHAVE dataset for the mesh-based interaction.
- [HO-GCN](https://hiwilliamwwl.github.io/Learn-to-Predict-How-Humans-Manipulate-Large-Sized-Objects-From-Interactive-Motions-objects/): We use its presented dataset for the skeleton-based interaction.
- [TEMOS](https://github.com/Mathux/TEMOS): We adopt the rendering code for HOI visualization.
- [MDM](https://guytevet.github.io/mdm-page/): We use the MDM in our work.
- [STARS](https://github.com/Sirui-Xu/STARS): We use the STARS in our work.

## 📚 License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, Hugging Face, Hydra, and uses datasets which each have their own respective licenses that must also be followed.

## 🌟 Star History

<p align="center">
    <a href="https://star-history.com/#Sirui-Xu/InterDiff&Date" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=Sirui-Xu/InterDiff&type=Date" alt="Star History Chart">
    </a>
<p>