---
title: From DFT to machine learning阅读笔记
author: 赵旭山
tags: 随笔
article_header:
  type: cover
  image:
    src: //assets/images/fromDFTtoMachineLearning202002232204.png
typora-root-url: ..
---

#### 1. From DFT to Machine Learning

DFT➞High-throughput approach➞Machine learning

![](/assets/images/fromDFTtoMLFigure1_202002232212.png)

#### 2. Fourth paradigm for scientific exploration

**Fourth paradigm**: (big) data-driven science

> ‘Originally, there was just experimental science, and then there was theoretical science, with Kepler’s Laws, Newton’s Laws of Motion, Maxwell’s equations, and so on. Then, for many problems, the theoretical models grew too complicated to solve analytically, and people had to start simulating. These simulations have carried us through much of the last half of the last century. At this point, these simulations are generating a whole lot of data, along with a huge increase in data from the exper- imental sciences. People now do not actually look through telescopes. Instead, they are ‘looking’ through large-scale, complex instruments which relay data to datacenters, and only then do they look at the information on their computers.
>
> The world of science has changed, and there is no question about this. The new model is for the data to be captured by instruments or generated by simulations before being processed by software and for the resulting information or knowledge to be stored in computers. Scientists only get to look at their data fairly late in this pipeline. The techniques and technologies for such data-intensive science are so different that it is worth distinguishing data-intensive science from computational science as a new, fourth paradigm for scientific exploration [4].’—Jim Gray, 2007 [5].

Experiment➞Theory➞Computation/Simulation➞Data-driven science

![](/assets/images/fromDFTtoMLFigure2_202002232227.png)

#### 3. Big Data characteristics known as the 'five V's'

**Domain Knowledge** & **Five V's**

* Related sixth V: **Visualization**

![](/assets/images/fromDFTtoMLFigure3_202002232300.png)

* Combination of **mathematics** and **statistics**, **computer science** and **programming**

  > It is largely interdisciplinary being a combination of **mathematics** and **statistics**, **computer science** and **programming**.

* Whole process of data: **production**, **cleaning**, **preparation**, **analysis**

  > Its objective is, roughly speaking, to deal with the whole process of data **production**, **cleaning**, **preparation**, and finally, **analysis**.

* **Knowledge Discovery in Databases (KDD)**

  > Data science encompasses areas such as Big Data, which deals with large volumes of data, and data mining, which relates to analysis processes to discover patterns and extract knowledge from data, part of the so-called **Knowledge Discovery in Databases (KDD)**.

#### 4. Three generation in computational materials science

* **1st Generation**: The first generation is related to **materials property attainment given its structure**, using local optimization algorithms, usually based on DFT calculations performed one at a time.

* **2nd Generation**: The second generation is related to **crystal structure prediction given a  fixed composition**, using global optimization tasks like genetic and evolutionary algorithms.

* **3rd Generation**: The third generation is based on **statistical learning**. It also enables the discovery of novel compositions, besides much faster predictions of properties and crystalline structures given the vast amount of available physical and chemical data via ML algorithms.

#### 5. Historical development of DFT

In 1964 Hohenberg and Kohn published an article that became the paradigm for the understanding of materials properties, today known as Density Functional Theory (DFT). 

$$ E[n]=T_s[n]+U_H[n]+V_{ext}[n]+E_{xc}[n] $$

$ E_{xc}[n] $ is  exchange-correlation term to the energy, $ T_s[n] $ is kinetic energy, $ U_H $ is the Hartree potential, $ V_{ext} $ is an external potential.

##### 5.1 All-electron treatment

First, one needs to select the exchange-correlation term contained in equation.

In the early days of DFT, only the so-called **all-electron treatment** was available, and its drawback was the restriction of systems that could be simulated at that time. 

##### 5.2 Pseudopotential & PAW

The pseudopotential method led to the possibility of simulation. Some popular approaches are the projector augmented waves (PAW), norm-conserving and ultrasoft pseudopotentials as developed by Troullier and Martins and Vanderbilt.

##### 5.3 Bloch's theorem

Another important advance in DFT was the treatment of materials imposing links on translational symmetry, via Bloch’s theorem.

##### 5.4 LDA & GGA

The pursuit of the ‘exact’ functional is still a subject of research. In its first implementation, DFT codes employed the Local Spin Density approximation (LSDA or simply LDA) for the exchange-correlation functional.

On the other hand, the chemistry community did not embrace LDA due to a few systematic errors, such as overestimation of molecular atomization energies and overestimation of bond lengths. Such shortcomings were alleviated in great part when the generalized gradient approximation (GGA) was introduced in the 1980s. 

A number of flavours of exchange-correlation functionals within this approximation are available, namely the Perdew- Burke-Ernzerhof (PBE), Perdew-Wang (PW91), and Becke-Lee-Yang-Parr (BLYP) are some examples of very successful functionals.

##### 5.5 meta-GGA

The next step in the complexity of exchange-correlation functionals is usually referred to as the advent of the meta-GGA approximation. Their new ingredient is the introduction of the so-called Kohn–Sham kinetic energy density $ \tau_{\uparrow / \downarrow}(r) $. opular functionals within this approximation comprise the Tao–Perdew–Staroverov–Scuseria functional (TPSS), and the more recent proposal of the non-empirical strongly constrained and appropriately normed (SCAN) functional of Sun et al. Successful attempts of semilocal functionals for improved bandgaps of different materials include the Tran–Blaha modified Becke–Johnson (mBJ) and ACBN0 functionals.

##### 5.6 HSE

Hybrid functional inspired by the Hartree–Fock formulation introduced non-locality in DFT by mixing a fraction of the exact exchange term into the exchange-correlation energy within the GGA. Examples are the PBE0 and the Coulomb interaction screened Heyd–Scuseria–Ernzerhof (HSE) hybrid functionals based on the PBE $ E_{xc} $ and the B3LYP functional, which introduced mixing as well as other empirical parameters into its precursor BLYP.

##### 5.7 RPA

Finally, by considering both occupied and unoccupied orbitals in the theory. Random Phase Approximation (RPA) can successfully account for electronic correlation.



#### 6. DFT calculations & Structure prediction: local minima & global optimization

> Base on the Hellman-Feynman theorem, one can use DFT calculations to find a local structural minima of materials and molecules. However, a global optimization of such systems is much more involved process.

#### 7. High-throughput (HT)

![](/assets/images/fromDFTtoMLFigure6_202002252015.png)