# biobeam - Fast simulation of image formation for in-silico tissue microscopy

<img src="https://github.com/maweigert/biobeam/raw/master/artwork/logo_biobeam_red.png" width="200">

*biobeam* is an open source python package for simulating light propagation on weakly scattering tissue. It implements a scalar beam propagation method while making use of GPU acceleration via PyOpenCL. It is designed to provide fast wave optical simulations common in light sheet microscopy while providing an easy to use API from within Python.

Among the features are

* Fast vectorial psf calculations for various illumination modes (gaussian.bessel beams, cylindrical lenses, bessel lattices...) 
* Fast scalar wave optical simulation of incident light interaction with weakly scattering tissue 
* Simulation module for image formation in light sheet microscopy and aberration calculations

### Quickstart

First make sure there is a valid OpenCL platform available on your machine, e.g. check with 

	clinfo
	
then install everything via pip  

	pip install biobeam


The full documentation with examples can be found [here](https://maweigert.github.io/biobeam).


最简洁的写法是：**用一套统一框架写，差异只体现在瞳面函数 $P(\theta, \phi)$ 的定义上**。

---

## 推荐结构

### 第一段：统一公式

先写 Richards-Wolf 矢量衍射积分作为所有 PSF 的共同基础：

$$\mathbf{E}(\mathbf{r}) = -\frac{i}{\lambda} \iint_\Omega P(\theta,\phi)\, \mathbf{A}(\theta,\phi)\, e^{i\mathbf{k}\cdot\mathbf{r}} \sin\theta \, d\theta \, d\phi$$

其中 $\mathbf{A}(\theta,\phi)$ 是矢量振幅因子（处理偏振），$P(\theta,\phi)$ 是瞳面函数，$\alpha = \arcsin(\mathrm{NA}/n)$ 是半锥角。强度 PSF 定义为 $h(\mathbf{r}) = |\mathbf{E}(\mathbf{r})|^2$。

---

### 第二段：四种显微镜 = 四种瞳面函数的组合

| 显微镜 | 瞳面函数 $P(\theta,\phi)$ | 有效 PSF |
|---|---|---|
| 宽场 | $P = 1,\; \theta \in [0, \alpha]$ | $h_\mathrm{WF} = h_\mathrm{em}$ |
| 共聚焦（小针孔极限） | 激发：$P=1,\;\theta\in[0,\alpha]$；发射：同左 | $h_\mathrm{conf} = h_\mathrm{ex} \cdot h_\mathrm{em}$ |
| SPIM | 照明：$P=1,\;\theta\in[0,\alpha_\mathrm{ill}]$（或环形孔 Bessel） | $h_\mathrm{SPIM} = h_\mathrm{ill} \cdot h_\mathrm{det}$ |
| Lattice Light-Sheet | 照明：$P(\theta,\phi) = \sum_i G(\phi-\phi_i)\cdot\mathbf{1}_{[\alpha_1,\alpha_2]}(\theta)$ | $h_\mathrm{LLS} = h_\mathrm{ill} \cdot h_\mathrm{det}$ |

SPIM 和 Lattice 的探测端 $h_\mathrm{det}$ 与宽场 $h_\mathrm{WF}$ 形式完全相同，只是使用发射波长。

---

## 这样写的好处

1. **只有一个公式**，所有差异都在 $P(\theta,\phi)$ 里，逻辑非常干净
2. 不需要提 BPM，也不需要区分 "Richards-Wolf" 和 "Debye-Wolf" 的名字——统一叫 Richards-Wolf，照明端只是换了瞳面函数
3. Lattice 的瞳面函数里的 $G(\phi - \phi_i)$ 是 Gaussian 模糊函数，$[\alpha_1, \alpha_2]$ 是内外环 NA 限定的角度范围，这样一行就把 Chen et al. 2014 的物理说清楚了
4. 审稿人很难质疑：每一步都有对应文献（Richards & Wolf 1959 打底，Chen 2014 给 Lattice 瞳面函数）