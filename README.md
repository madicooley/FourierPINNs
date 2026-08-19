# FourierPINNs

This repository contains the code accompanying the paper **“Fourier PINNs: From Strong Boundary Conditions to Adaptive Fourier Bases”** by Madison Cooley, Varun Shankar, Robert M. Kirby, and Shandian Zhe.

## Abstract
Interest is rising in Physics-Informed Neural Networks (PINNs) as a mesh-free alternative to traditional numerical solvers for partial differential equations (PDEs). 
However, PINNs often struggle to learn high-frequency and multi-scale target solutions. 
To tackle this problem, we first study a strong Boundary Condition (BC) version of PINNs for Dirichlet BCs and observe a consistent decline in relative error compared to the standard PINNs. 
We then perform a theoretical analysis based on the Fourier transform and convolution theorem. 
We find that strong BC PINNs can better learn the amplitudes of high-frequency components of the target solutions. 
However, constructing the architecture for strong BC PINNs is difficult for many BCs and domain geometries. 
Enlightened by our theoretical analysis, we propose Fourier PINNs --- a simple, general, yet powerful method that augments PINNs with pre-specified, dense Fourier bases. 
Our proposed architecture likewise learns high-frequency components better but places no restrictions on the particular BCs or problem domains. 
We develop an adaptive learning and basis selection algorithm via alternating neural net basis optimization, Fourier and neural net basis coefficient estimation, and coefficient truncation. 
This scheme can flexibly identify the significant frequencies while weakening the nominal frequencies to better capture the target solution's power spectrum. 
We show the advantage of our approach through a set of systematic experiments.  



---

## Contact

For questions about this work, please email mcooley220 [at] gmail [dot] com.
