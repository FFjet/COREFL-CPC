# Two-Temperature Inviscid-Flux Jacobian and Characteristic WENO Audit

This note gives the full characteristic derivation used by the finite-difference WENO path in `src/WENO.cu`, including the two-temperature scalar `rhoEve`.  It is written against the conservative-variable layout used by COREFL and the inviscid-flux/eigenvector derivation in `化学反应流动无粘通量雅可比矩阵及其特征向量.pdf`.  The repository copy of `无粘通量离散.pdf` is damaged beyond its first recovered page, but that page identifies the intended algorithm: use a Roe-averaged interface state, project a local Lax-Friedrichs flux split to characteristic space, reconstruct by WENO, then project back to conservative fluxes.

Line numbers below refer to the current tree when this document was written.

## Code Path

The WENO entry point is:

| Item | Code |
| --- | --- |
| Select WENO inviscid discretization | `src/InviscidScheme.cu:18-20` calls `compute_convective_term_weno` when `inviscid_type = 3`. |
| Launch x/y/z WENO flux kernels and derivative kernels | `src/WENO.cu:2888-2920`. |
| Component WENO, not characteristic | `inviscid_scheme = 51` or `71`, handled in `src/WENO.cu:890-989`. |
| Characteristic WENO5/WENO7 | `inviscid_scheme = 52` or `72`, handled after `src/WENO.cu:1090`. |
| Shock sensor switch | `src/WENO.cu:839-849`; if `if_shock=false`, WENO routines use the optimal linear weights. |
| Final finite-difference residual | `src/WENO.cu:1551-1558`, `src/WENO.cu:2215-2222`, `src/WENO.cu:2878-2885`: `dq -= F_{i+1/2}-F_{i-1/2}`. |

For two-temperature finite-rate mixture runs, `rhoEve` is one transported scalar:

| Variable | Code |
| --- | --- |
| `i_eve = n_spec` in scalar storage | `src/Parameter.cpp:647-655`. |
| `rhoEve` conservative index `i_eve_cv = 5+n_spec` | `src/Parameter.cpp:647-650`. |
| Output scalar `Eve` and output-only `Tve` | `src/Parameter.cpp:864-868`, `src/Parameter.cpp:914-917`. |
| Number of WENO transported scalars | `param->n_scalar_transported`, updated in `src/Parameter.cpp:647-676`. |

## Conservative Variables

COREFL stores one mixture density plus transported scalars.  For characteristic WENO, define

$$
Q =
\begin{bmatrix}
\rho \\
\rho u \\
\rho v \\
\rho w \\
\rho E \\
\rho\phi_1 \\
\vdots \\
\rho\phi_M
\end{bmatrix},
\qquad
\phi_a \in \{Y_1,\ldots,Y_{N_s},E_{ve},\text{other transported scalars}\}.
$$

Here

$$
M = \texttt{param->n\_scalar\_transported}.
$$

For a two-temperature AIR5 case without turbulence/passive scalars,

$$
\phi_a =
\begin{cases}
Y_a, & 0 \le a < N_s, \\
E_{ve}, & a = i_{eve}=N_s .
\end{cases}
$$

The two-temperature total energy is

$$
E = K + e_{tr}(T,Y) + E_{ve},
\qquad
K = \frac12(u^2+v^2+w^2),
$$

with

$$
h_{tr,s}(T) = h_{eq,s}(T) - E_{ve,s}(T),
\qquad
e_{tr,s}(T) = h_{tr,s}(T) - R_s T,
$$

and

$$
e_{tr}(T,Y) = \sum_s Y_s e_{tr,s}(T).
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| `E = K + e_tr + Eve` for reconstructed primitive states | `src/InviscidScheme.cu:184-209`. |
| `h_tr,s = h_eq,s - E_ve,s` | `src/Thermo.cuh:62-68`. |
| `e_tr = sum Y_s(h_tr,s-R_sT)` | `src/Thermo.cuh:57-72`. |

## Curvilinear Inviscid Flux

Let the metric direction be

$$
\boldsymbol{k}=(k_x,k_y,k_z)^T,\qquad
|\boldsymbol{k}|=\sqrt{k_x^2+k_y^2+k_z^2},\qquad
\boldsymbol{n}=\frac{\boldsymbol{k}}{|\boldsymbol{k}|}.
$$

Define the contravariant velocity

$$
U_k = \boldsymbol{k}\cdot\boldsymbol{u}
    = k_xu+k_yv+k_zw,
\qquad
U_n = \boldsymbol{n}\cdot\boldsymbol{u}.
$$

The inviscid flux in direction $\boldsymbol{k}$ is

$$
F_k(Q)=
\begin{bmatrix}
\rho U_k \\
\rho u U_k + k_xp \\
\rho v U_k + k_yp \\
\rho w U_k + k_zp \\
(\rho E+p)U_k \\
\rho\phi_1 U_k \\
\vdots \\
\rho\phi_M U_k
\end{bmatrix}.
$$

The two-temperature scalar therefore has

$$
F_{E_{ve}}^{inv} = \rho E_{ve} U_k.
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| `U_k=(q1*kx+q2*ky+q3*kz)/rho` | `src/WENO.cu:1113-1117`. |
| momentum fluxes `rho u_i U_k + k_i p` | `src/WENO.cu:1120-1122`. |
| pressure temporary `F[4]=p` | `src/WENO.cu:1117-1124`. |
| scalar flux is not stored separately; it is `Q[5+n]*F[0]` | `src/WENO.cu:1226-1227`, `src/WENO.cu:1307-1319`. |
| local Jacobian factor `J` stored as `F[6]` | `src/WENO.cu:1137`. |

## Pressure Differential

The characteristic projection needs $\partial p/\partial Q$ for the frozen inviscid system.  Start from

$$
\rho e_{tr}
=
\rho E - \frac{(\rho u)^2+(\rho v)^2+(\rho w)^2}{2\rho}
- \rho E_{ve}.
$$

Differentiating the kinetic term gives

$$
dK_\rho
=
d\!\left(
\frac{(\rho u)^2+(\rho v)^2+(\rho w)^2}{2\rho}
\right)
=
u\,d(\rho u)+v\,d(\rho v)+w\,d(\rho w)-K\,d\rho.
$$

Therefore

$$
d(\rho e_{tr})
=
d(\rho E)
-u\,d(\rho u)-v\,d(\rho v)-w\,d(\rho w)
+K\,d\rho
-d(\rho E_{ve}).
$$

The thermodynamic differential of the translational-rotational internal energy is

$$
d(\rho e_{tr})
=
\rho c_{v,tr}\,dT + \sum_s e_{tr,s}(T)\,d(\rho Y_s).
$$

The equation of state is

$$
p = T\sum_s \rho Y_s R_s,
$$

so

$$
dp
=
\left(\sum_s \rho Y_s R_s\right)dT
+T\sum_s R_s\,d(\rho Y_s).
$$

Use

$$
R_{mix}=\sum_s Y_sR_s,\qquad
\Gamma=\gamma_{tr}-1=\frac{R_{mix}}{c_{v,tr}},
\qquad
c^2=\gamma_{tr}R_{mix}T,
$$

and eliminate $dT$:

$$
\begin{aligned}
dp
&=
\Gamma\left[
d(\rho E)
-u\,d(\rho u)-v\,d(\rho v)-w\,d(\rho w)
+K\,d\rho
-d(\rho E_{ve})
-\sum_s e_{tr,s}\,d(\rho Y_s)
\right]
\\
&\quad + T\sum_s R_s\,d(\rho Y_s).
\end{aligned}
$$

Collecting the species coefficients,

$$
R_sT-\Gamma e_{tr,s}
=
R_sT-\Gamma(h_{tr,s}-R_sT)
=
\gamma_{tr}R_sT-\Gamma h_{tr,s}.
$$

Thus the final conservative pressure differential is

$$
\boxed{
\begin{aligned}
dp
&=
\Gamma\left[
K\,d\rho
-u\,d(\rho u)-v\,d(\rho v)-w\,d(\rho w)
+d(\rho E)
-d(\rho E_{ve})
\right]
\\
&\quad
+\sum_s
\left(
\gamma_{tr}R_sT-\Gamma h_{tr,s}
\right)d(\rho Y_s).
\end{aligned}
}
$$

It is convenient to define one scalar pressure coefficient $\alpha_a$ for every transported scalar:

$$
dp =
\Gamma K\,d\rho
-\Gamma u\,d(\rho u)
-\Gamma v\,d(\rho v)
-\Gamma w\,d(\rho w)
+\Gamma\,d(\rho E)
+c^2\sum_{a=1}^{M}\alpha_a\,d(\rho\phi_a),
$$

where

$$
\alpha_s =
\frac{\gamma_{tr}R_sT-\Gamma h_{tr,s}(T)}{c^2},
\qquad
\alpha_{E_{ve}} =
-\frac{\Gamma}{c^2},
\qquad
\alpha_a=0
\text{ for pressure-passive scalars}.
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| `Gamma = gamma_local - 1` | `src/Thermo.cuh:95`. |
| `c^2 = gamma_local*R_mix*T` | `src/Thermo.cuh:70-76`. |
| species `alpha_s` | `src/Thermo.cuh:99-103`. |
| `alpha_Eve = -Gamma/c^2` | `src/Thermo.cuh:106-108`. |
| pressure-passive scalars default to zero | `src/Thermo.cuh:87-93`. |

## Conservative Jacobian

Let

$$
q_0=\rho,\quad
\boldsymbol{m}=(q_1,q_2,q_3)^T=\rho\boldsymbol{u},\quad
q_4=\rho E,\quad
q_{5+a}=\rho\phi_a.
$$

Define

$$
p_0=\frac{\partial p}{\partial q_0}=\Gamma K,
\quad
p_i=\frac{\partial p}{\partial q_i}=-\Gamma u_i\quad (i=1,2,3),
\quad
p_4=\Gamma,
\quad
p_{5+a}=c^2\alpha_a.
$$

The conservative Jacobian

$$
A_k=\frac{\partial F_k}{\partial Q}
$$

has the following compact full form:

$$
A_k =
\begin{bmatrix}
0 & k_x & k_y & k_z & 0 & 0 & \cdots & 0 \\
-uU_k+k_xp_0 & U_k+uk_x+k_xp_1 & uk_y+k_xp_2 & uk_z+k_xp_3 & k_xp_4 & k_xp_{5+1} & \cdots & k_xp_{5+M} \\
-vU_k+k_yp_0 & vk_x+k_yp_1 & U_k+vk_y+k_yp_2 & vk_z+k_yp_3 & k_yp_4 & k_yp_{5+1} & \cdots & k_yp_{5+M} \\
-wU_k+k_zp_0 & wk_x+k_zp_1 & wk_y+k_zp_2 & U_k+wk_z+k_zp_3 & k_zp_4 & k_zp_{5+1} & \cdots & k_zp_{5+M} \\
U_kp_0-HU_k & Hk_x+U_kp_1 & Hk_y+U_kp_2 & Hk_z+U_kp_3 & U_k(1+p_4) & U_kp_{5+1} & \cdots & U_kp_{5+M} \\
-\phi_1U_k & \phi_1k_x & \phi_1k_y & \phi_1k_z & 0 & U_k & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
-\phi_MU_k & \phi_Mk_x & \phi_Mk_y & \phi_Mk_z & 0 & 0 & \cdots & U_k
\end{bmatrix},
$$

where

$$
H=E+\frac{p}{\rho}.
$$

This matrix is not explicitly assembled in `WENO.cu`; the code applies its eigenvectors directly.

## Primitive Matrix and Eigenvalues

Use primitive variables

$$
W=(\rho,u,v,w,p,\phi_1,\ldots,\phi_M)^T.
$$

For the homogeneous inviscid part, define the directional material operator

$$
D_k(\cdot)=\partial_t(\cdot)+U_k\partial_\xi(\cdot).
$$

Expanding the conservative equations gives the primitive system used for the eigenvalue derivation.  The continuity equation is

$$
D_k\rho+\rho\left(k_x\partial_\xi u+k_y\partial_\xi v+k_z\partial_\xi w\right)=0.
$$

Subtracting velocity times the continuity equation from each momentum equation gives

$$
D_ku+\frac{k_x}{\rho}\partial_\xi p=0,
\qquad
D_kv+\frac{k_y}{\rho}\partial_\xi p=0,
\qquad
D_kw+\frac{k_z}{\rho}\partial_\xi p=0.
$$

The frozen thermodynamic pressure equation follows from the pressure differential above and the scalar equations

$$
D_k\phi_a=0.
$$

Because scalar gradients are advected by the scalar equations, the pressure equation reduces to

$$
D_kp+\rho c^2
\left(k_x\partial_\xi u+k_y\partial_\xi v+k_z\partial_\xi w\right)=0.
$$

The frozen inviscid primitive system in direction $\boldsymbol{k}$ is

$$
\partial_t W + A_W\partial_\xi W=0,
$$

with

$$
A_W=
\begin{bmatrix}
U_k & \rho k_x & \rho k_y & \rho k_z & 0 & 0 & \cdots & 0 \\
0 & U_k & 0 & 0 & k_x/\rho & 0 & \cdots & 0 \\
0 & 0 & U_k & 0 & k_y/\rho & 0 & \cdots & 0 \\
0 & 0 & 0 & U_k & k_z/\rho & 0 & \cdots & 0 \\
0 & \rho c^2k_x & \rho c^2k_y & \rho c^2k_z & U_k & 0 & \cdots & 0 \\
0 & 0 & 0 & 0 & 0 & U_k & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & 0 & 0 & 0 & \cdots & U_k
\end{bmatrix}.
$$

The determinant is

$$
\det(A_W-\lambda I)
=
(U_k-\lambda)^{M+3}
\left[(U_k-\lambda)^2-c^2|\boldsymbol{k}|^2\right].
$$

Therefore the eigenvalues of the conservative Jacobian are

$$
\boxed{
\lambda_- = U_k-c|\boldsymbol{k}|,
\qquad
\lambda_0 = U_k \quad \text{with multiplicity } M+3,
\qquad
\lambda_+ = U_k+c|\boldsymbol{k}|.
}
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| pointwise spectral radius `|U_k|+c|k|` | `src/WENO.cu:1129-1136`. |
| stencil maximum for LF splitting, WENO7 | `src/WENO.cu:1212-1218`, `src/WENO.cu:1943-1949`, `src/WENO.cu:2606-2612`. |
| stencil maximum for LF splitting, WENO5 | `src/WENO.cu:1326-1332`, `src/WENO.cu:2061-2067`, `src/WENO.cu:2724-2730`. |

## Roe-Averaged Interface State

At interface $i+\frac12$, define

$$
d_L=\sqrt{\rho_L},\qquad
d_R=\sqrt{\rho_R},\qquad
\omega_L=\frac{d_L}{d_L+d_R},\qquad
\omega_R=\frac{d_R}{d_L+d_R}.
$$

The Roe-averaged velocity and scalars are

$$
\tilde{\boldsymbol{u}}=\omega_L\boldsymbol{u}_L+\omega_R\boldsymbol{u}_R,
\qquad
\tilde{\phi}_a=\omega_L\phi_{a,L}+\omega_R\phi_{a,R}.
$$

In WENO code, `Q` stores conservative values, so the algebra is written as

$$
\omega_L\phi_{a,L}
=
\frac{\rho_L\phi_{a,L}}{\rho_L+\sqrt{\rho_L\rho_R}},
\qquad
\omega_R\phi_{a,R}
=
\frac{\rho_R\phi_{a,R}}{\rho_R+\sqrt{\rho_L\rho_R}}.
$$

The temperature used for $\gamma_{tr}$, $c$, $\alpha_a$, and scalar energy coefficients is estimated as

$$
\tilde{T}
=
\frac{
\omega_L(p_L/\rho_L)+\omega_R(p_R/\rho_R)
}{
\tilde{R}_{mix}
},
\qquad
\tilde{R}_{mix}=\sum_s \tilde{Y}_sR_s.
$$

The interface metric is averaged and then normalized for the eigenvectors:

$$
\tilde{\boldsymbol{k}}
=
\frac12(\boldsymbol{k}_L+\boldsymbol{k}_R),
\qquad
\boldsymbol{n}=\frac{\tilde{\boldsymbol{k}}}{|\tilde{\boldsymbol{k}}|}.
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| `sqrt(rhoL*rhoR)` and conservative-weight coefficients | `src/WENO.cu:1146-1151`, `src/WENO.cu:1875-1879`, `src/WENO.cu:2538-2542`. |
| Roe velocity | `src/WENO.cu:1151-1153`, `src/WENO.cu:1880-1882`, `src/WENO.cu:2543-2545`. |
| Roe transported scalars `svm[l]` | `src/WENO.cu:1155-1158`, `src/WENO.cu:1884-1887`, `src/WENO.cu:2547-2550`. |
| temperature estimate | `src/WENO.cu:1160-1164`, `src/WENO.cu:1889-1894`, `src/WENO.cu:2552-2557`. |
| thermodynamic characteristic coefficients | `src/WENO.cu:1166-1173`, `src/WENO.cu:1896-1903`, `src/WENO.cu:2559-2566`. |
| normalize metric and compute normal velocity | `src/WENO.cu:1175-1180`, `src/WENO.cu:1905-1910`, `src/WENO.cu:2568-2573`. |

## Tangential Basis

The right and left eigenvectors use an orthonormal triad

$$
\{\boldsymbol{n},\boldsymbol{t},\boldsymbol{b}\},
\qquad
\boldsymbol{n}\cdot\boldsymbol{t}
=\boldsymbol{n}\cdot\boldsymbol{b}
=\boldsymbol{t}\cdot\boldsymbol{b}=0.
$$

The code calls the normal components `kx, ky, kz`, the first tangent `nx, ny, nz`, and the second tangent `qx, qy, qz`.  Mathematically in this note:

$$
\boldsymbol{n}=(n_x,n_y,n_z),\qquad
\boldsymbol{t}=(t_x,t_y,t_z),\qquad
\boldsymbol{b}=(b_x,b_y,b_z).
$$

Define

$$
U_n=\boldsymbol{n}\cdot\tilde{\boldsymbol{u}},
\qquad
U_t=\boldsymbol{t}\cdot\tilde{\boldsymbol{u}},
\qquad
U_b=\boldsymbol{b}\cdot\tilde{\boldsymbol{u}}.
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| choose first tangent robustly | `src/WENO.cu:1181-1192`, `src/WENO.cu:1911-1922`, `src/WENO.cu:2574-2585`. |
| second tangent by cross product | `src/WENO.cu:1193-1199`, `src/WENO.cu:1923-1929`, `src/WENO.cu:2586-2592`. |
| tangential velocities | `src/WENO.cu:1200-1201`, `src/WENO.cu:1930-1931`, `src/WENO.cu:2593-2594`. |

## Right Eigenvectors

Let

$$
\tilde{H}=\tilde{E}+\frac{\tilde{p}}{\tilde{\rho}},
\qquad
\Gamma=\tilde{\gamma}_{tr}-1.
$$

The scalar eigenvectors must keep $dp=0$.  For scalar $a$, choose the total-energy entry

$$
\chi_a = -\frac{c^2\alpha_a}{\Gamma}.
$$

Thus

$$
\chi_s =
-\frac{c^2\alpha_s}{\Gamma}
\quad \text{for species},
\qquad
\chi_{E_{ve}} = 1,
\qquad
\chi_a=0
\quad \text{for pressure-passive scalars}.
$$

In conservative variables ordered as

$$
[\rho,\rho u,\rho v,\rho w,\rho E,\rho\phi_1,\ldots,\rho\phi_M]^T,
$$

a complete right eigenbasis is:

$$
r_- =
\begin{bmatrix}
1\\
u-cn_x\\
v-cn_y\\
w-cn_z\\
H-cU_n\\
\phi_1\\
\vdots\\
\phi_M
\end{bmatrix},
\qquad
r_t =
\begin{bmatrix}
0\\
t_x\\
t_y\\
t_z\\
U_t\\
0\\
\vdots\\
0
\end{bmatrix},
$$

$$
r_b =
\begin{bmatrix}
0\\
b_x\\
b_y\\
b_z\\
U_b\\
0\\
\vdots\\
0
\end{bmatrix},
\qquad
r_c =
\begin{bmatrix}
1\\
u\\
v\\
w\\
H-\frac{c^2}{\Gamma}\\
\phi_1\\
\vdots\\
\phi_M
\end{bmatrix},
$$

$$
r_+ =
\begin{bmatrix}
1\\
u+cn_x\\
v+cn_y\\
w+cn_z\\
H+cU_n\\
\phi_1\\
\vdots\\
\phi_M
\end{bmatrix},
\qquad
r_{\phi_a} =
\begin{bmatrix}
0\\
0\\
0\\
0\\
\chi_a\\
0\\
\vdots\\
1\text{ at scalar }a\\
\vdots\\
0
\end{bmatrix}.
$$

The two-temperature scalar eigenvector has

$$
r_{E_{ve}} =
[0,0,0,0,1,0,\ldots,1\text{ at }\rho E_{ve},\ldots,0]^T.
$$

This means a frozen scalar-wave perturbation in `rhoEve` also changes total energy by the same amount, leaving translational pressure unchanged.

Code correspondence:

| Formula | Code |
| --- | --- |
| scalar right-eigenvector energy coefficients `chi_a` | `src/Thermo.cuh:101-108`. |
| `chi_Eve=1` | `src/Thermo.cuh:106-108`. |
| right multiplication is written explicitly, not as a matrix | `src/WENO.cu:1439-1457`, `src/WENO.cu:2174-2192`, `src/WENO.cu:2837-2855`. |

## Left Eigenvectors

For an arbitrary conservative perturbation or flux-like vector $z$, write

$$
z =
[z_\rho,z_{\rho u},z_{\rho v},z_{\rho w},z_{\rho E},z_{\rho\phi_1},\ldots,z_{\rho\phi_M}]^T.
$$

Define

$$
\beta(z)=\sum_{a=1}^{M}\alpha_a z_{\rho\phi_a}.
$$

The left eigenvectors dual to the right basis above are:

$$
L_-z
=
\frac12\left[
\frac{\Gamma K+cU_n}{c^2}z_\rho
-\frac{(\Gamma\boldsymbol{u}+c\boldsymbol{n})\cdot\boldsymbol{z}_m}{c^2}
+\frac{\Gamma}{c^2}z_{\rho E}
+\beta(z)
\right],
$$

$$
L_tz=-U_tz_\rho+\boldsymbol{t}\cdot\boldsymbol{z}_m,
\qquad
L_bz=-U_bz_\rho+\boldsymbol{b}\cdot\boldsymbol{z}_m,
$$

$$
L_cz
=
\left(1-\frac{\Gamma K}{c^2}\right)z_\rho
+\frac{\Gamma}{c^2}\boldsymbol{u}\cdot\boldsymbol{z}_m
-\frac{\Gamma}{c^2}z_{\rho E}
-\beta(z),
$$

$$
L_+z
=
\frac12\left[
\frac{\Gamma K-cU_n}{c^2}z_\rho
-\frac{(\Gamma\boldsymbol{u}-c\boldsymbol{n})\cdot\boldsymbol{z}_m}{c^2}
+\frac{\Gamma}{c^2}z_{\rho E}
+\beta(z)
\right],
$$

and for every transported scalar,

$$
L_{\phi_a}z=z_{\rho\phi_a}-\phi_a z_\rho.
$$

These rows satisfy

$$
LR=I.
$$

The cancellation for scalar waves follows directly:

$$
L_-r_{\phi_a}
=
\frac12\left(
\frac{\Gamma}{c^2}\chi_a+\alpha_a
\right)
=0,
\qquad
L_cr_{\phi_a}
=
-\frac{\Gamma}{c^2}\chi_a-\alpha_a
=0.
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| `Gamma*K` as `temp3=0.5*gm1*(u^2+v^2+w^2)` | `src/WENO.cu:1207-1209`, `src/WENO.cu:1937-1939`, `src/WENO.cu:2600-2602`. |
| `1/c^2` as `temp2` | `src/WENO.cu:1203-1205`, `src/WENO.cu:1933-1935`, `src/WENO.cu:2596-2598`. |
| acoustic minus row `L_-` | `src/WENO.cu:1237-1243`, `src/WENO.cu:1968-1974`, `src/WENO.cu:2631-2637`. |
| first tangent row `L_t` | `src/WENO.cu:1244-1251`, `src/WENO.cu:1975-1982`, `src/WENO.cu:2638-2645`. |
| second tangent row `L_b` | `src/WENO.cu:1252-1259`, `src/WENO.cu:1983-1990`, `src/WENO.cu:2646-2653`. |
| contact row `L_c` | `src/WENO.cu:1260-1267`, `src/WENO.cu:1991-1998`, `src/WENO.cu:2654-2661`. |
| acoustic plus row `L_+` | `src/WENO.cu:1268-1274`, `src/WENO.cu:1999-2005`, `src/WENO.cu:2662-2668`. |
| `beta(Q)=sum alpha_a Q_a` and `beta(F)=sum alpha_a F_a` | `src/WENO.cu:1220-1230`, `src/WENO.cu:1334-1344`, `src/WENO.cu:1951-1961`, `src/WENO.cu:2069-2079`, `src/WENO.cu:2614-2624`, `src/WENO.cu:2732-2742`. |
| scalar rows `Q_a - phi_a Q_0` and `F_a - phi_a F_0` | `src/WENO.cu:1304-1321`, `src/WENO.cu:1418-1435`, `src/WENO.cu:2039-2056`, `src/WENO.cu:2153-2170`, `src/WENO.cu:2702-2719`, `src/WENO.cu:2816-2833`. |

## Characteristic Local Lax-Friedrichs Splitting

At each WENO stencil point $j$, the code uses the fixed interface left matrix $\tilde{L}$, but the pointwise flux and conservative state:

$$
\mathcal{F}_j = F_k(Q_j),
\qquad
Q_j=Q(x_j).
$$

The local Lax-Friedrichs split in characteristic space is

$$
\boxed{
g_j^\pm
=
\frac12 J_j
\left[
\tilde{L}\mathcal{F}_j
\pm
\lambda_{\max}\tilde{L}Q_j
\right],
}
$$

where

$$
\lambda_{\max}
=
\max_{j\in\text{stencil}}
\left(
|U_{k,j}|+c_j|\boldsymbol{k}_j|
\right).
$$

The $J_j$ factor is included before WENO reconstruction because the finite-difference update differentiates the transformed flux $JF$.

For rows $L_-$, $L_c$, and $L_+$, the scalar pressure coupling is inserted through

$$
\beta(Q_j)=\sum_a\alpha_a(\rho\phi_a)_j,
\qquad
\beta(\mathcal{F}_j)=
\sum_a\alpha_a(\rho\phi_aU_k)_j.
$$

For scalar rows,

$$
\tilde{L}_{\phi_a}Q_j=(\rho\phi_a)_j-\tilde{\phi}_a\rho_j,
$$

and

$$
\tilde{L}_{\phi_a}\mathcal{F}_j
=
(\rho\phi_aU_k)_j-\tilde{\phi}_a(\rho U_k)_j
=
U_{k,j}\left[(\rho\phi_a)_j-\tilde{\phi}_a\rho_j\right].
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| WENO7 stencil base and `lambda_max` | `src/WENO.cu:1210-1218`, `src/WENO.cu:1940-1949`, `src/WENO.cu:2603-2612`. |
| WENO5 stencil base and `lambda_max` | `src/WENO.cu:1323-1332`, `src/WENO.cu:2058-2067`, `src/WENO.cu:2721-2730`. |
| nonscalar `L F` and `L Q` | `src/WENO.cu:1282-1301`, `src/WENO.cu:1396-1415`, `src/WENO.cu:2013-2037`, `src/WENO.cu:2131-2151`, `src/WENO.cu:2676-2700`, `src/WENO.cu:2794-2814`. |
| scalar `L F` and `L Q` | same scalar-row ranges listed above. |
| multiply by `0.5*J_j` | `F[6][iP]` in `src/WENO.cu:1286-1301`, `src/WENO.cu:1400-1415`, and corresponding y/z ranges. |

## WENO Reconstruction

The reconstructed characteristic flux at the interface is

$$
\hat{g}_{i+1/2}
=
\mathcal{R}^+(g^+)
+
\mathcal{R}^-(g^-).
$$

In code this is stored as `fChar[l]`.

### WENO5

For the plus part, let the five input values be

$$
f_0,f_1,f_2,f_3,f_4
\equiv
\texttt{vp[0]},\ldots,\texttt{vp[4]}.
$$

The three candidate polynomials are

$$
p_0=\frac{2f_2+5f_3-f_4}{6},
\qquad
p_1=\frac{-f_1+5f_2+2f_3}{6},
\qquad
p_2=\frac{2f_0-7f_1+11f_2}{6}.
$$

The smoothness indicators are

$$
\beta_0 =
\frac{13}{12}(f_2-2f_3+f_4)^2
+\frac14(3f_2-4f_3+f_4)^2,
$$

$$
\beta_1 =
\frac{13}{12}(f_1-2f_2+f_3)^2
+\frac14(f_1-f_3)^2,
$$

$$
\beta_2 =
\frac{13}{12}(f_0-2f_1+f_2)^2
+\frac14(f_0-4f_1+3f_2)^2.
$$

In shocked regions (`if_shock=true`) the code uses Jiang-Shu nonlinear weights

$$
a_r=\frac{d_r}{(\epsilon+\beta_r)^2},
\qquad
\omega_r=\frac{a_r}{a_0+a_1+a_2},
\qquad
(d_0,d_1,d_2)=\left(\frac{3}{10},\frac{6}{10},\frac{1}{10}\right).
$$

Thus

$$
\mathcal{R}^+(f)
=
\omega_0p_0+\omega_1p_1+\omega_2p_2.
$$

In smooth regions (`if_shock=false`),

$$
\mathcal{R}^+(f)
=
\frac{3}{10}p_0+\frac{6}{10}p_1+\frac{1}{10}p_2.
$$

The minus reconstruction uses the reversed-biased formulas in `vm[0..4]` and optimal weights

$$
\left(\frac{1}{10},\frac{6}{10},\frac{3}{10}\right).
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| WENO5 candidate polynomials and smoothness indicators | `src/WENO.cu:3033-3074`. |
| WENO5 linear fallback | `src/WENO.cu:3076-3087`. |
| characteristic WENO5 calls | `src/WENO.cu:1416`, `src/WENO.cu:1435`, `src/WENO.cu:2151`, `src/WENO.cu:2170`, `src/WENO.cu:2814`, `src/WENO.cu:2833`. |

### WENO7

For the plus part, let

$$
f_0,\ldots,f_6 \equiv \texttt{vp[0]},\ldots,\texttt{vp[6]}.
$$

The four candidate polynomials are

$$
p_0=\frac{-3f_0+13f_1-23f_2+25f_3}{12},
$$

$$
p_1=\frac{f_1-5f_2+13f_3+3f_4}{12},
$$

$$
p_2=\frac{-f_2+7f_3+7f_4-f_5}{12},
$$

$$
p_3=\frac{3f_3+13f_4-5f_5+f_6}{12}.
$$

The optimal weights are

$$
(d_0,d_1,d_2,d_3)
=
\left(
\frac{1}{35},
\frac{12}{35},
\frac{18}{35},
\frac{4}{35}
\right).
$$

The code computes the seventh-order smoothness indicators through derivative-like quantities.  For stencil $r$,

$$
\beta_r
=
s_{1,r}^2
+\frac{13}{12}s_{2,r}^2
+\frac{1043}{960}s_{3,r}^2
+\frac{1}{12}s_{1,r}s_{3,r},
$$

with the $s_{m,r}$ definitions exactly as coded in `src/WENO.cu:3096-3117`.  The global WENO-Z indicator is

$$
\tau_7^2
=
(\beta_0+3\beta_1-3\beta_2-\beta_3)^2.
$$

In shocked regions, the nonlinear weights are

$$
a_r
=
d_r
+
d_r\frac{\tau_7^2}{(\epsilon+\beta_r)^2},
\qquad
\omega_r=\frac{a_r}{\sum_{m=0}^{3}a_m}.
$$

Thus

$$
\mathcal{R}^+(f)=\sum_{r=0}^{3}\omega_r p_r.
$$

In smooth regions, the code uses the linear optimal form

$$
\mathcal{R}^+(f)=\sum_{r=0}^{3}d_r p_r.
$$

The minus reconstruction uses the reversed-biased `vm[0..6]` formulas in the second half of `WENO7`.

Code correspondence:

| Formula | Code |
| --- | --- |
| WENO7 plus smoothness indicators | `src/WENO.cu:3090-3138`. |
| WENO7 minus smoothness indicators and reversed candidates | `src/WENO.cu:3139-3177`. |
| WENO7 linear fallback | `src/WENO.cu:3179-3195`. |
| characteristic WENO7 calls | `src/WENO.cu:1302`, `src/WENO.cu:1321`, `src/WENO.cu:2037`, `src/WENO.cu:2056`, `src/WENO.cu:2700`, `src/WENO.cu:2719`. |

## Back Projection

Let the reconstructed characteristic fluxes be

$$
a_-=\texttt{fChar[0]},
\quad
a_t=\texttt{fChar[1]},
\quad
a_b=\texttt{fChar[2]},
\quad
a_c=\texttt{fChar[3]},
\quad
a_+=\texttt{fChar[4]},
$$

and

$$
a_{\phi_a}=\texttt{fChar[5+a]}.
$$

Define

$$
S=a_-+a_c+a_+,
\qquad
D=a_- - a_+.
$$

Multiplication by the right eigenvectors gives

$$
\hat{F}_\rho = S,
$$

$$
\hat{\boldsymbol{F}}_m
=
\tilde{\boldsymbol{u}}S
-c\boldsymbol{n}D
+\boldsymbol{t}a_t
+\boldsymbol{b}a_b,
$$

$$
\hat{F}_{\rho E}
=
\tilde{H}S
-cU_nD
+U_ta_t
+U_ba_b
-\frac{c^2}{\Gamma}a_c
+\sum_a\chi_a a_{\phi_a},
$$

and

$$
\hat{F}_{\rho\phi_a}
=
\tilde{\phi}_aS+a_{\phi_a}.
$$

For `rhoEve`, $\chi_{E_{ve}}=1$, so the scalar characteristic flux contributes directly to the reconstructed total-energy flux:

$$
\Delta \hat{F}_{\rho E}^{(E_{ve})}=a_{E_{ve}}.
$$

Code correspondence:

| Formula | Code |
| --- | --- |
| `S=fChar[0]+fChar[3]+fChar[4]` | `src/WENO.cu:1441`, `src/WENO.cu:2176`, `src/WENO.cu:2839`. |
| `D=fChar[0]-fChar[4]` | `src/WENO.cu:1442`, `src/WENO.cu:2177`, `src/WENO.cu:2840`. |
| density and momentum back projection | `src/WENO.cu:1443-1446`, `src/WENO.cu:2178-2181`, `src/WENO.cu:2841-2844`. |
| Roe enthalpy `H` | `src/WENO.cu:1448-1449`, `src/WENO.cu:2183-2184`, `src/WENO.cu:2846-2847`. |
| total energy without scalar eigenvectors | `src/WENO.cu:1450`, `src/WENO.cu:2185`, `src/WENO.cu:2848`. |
| scalar fluxes `phi*S+a_phi` | `src/WENO.cu:1452-1455`, `src/WENO.cu:2187-2190`, `src/WENO.cu:2850-2853`. |
| scalar energy contributions `sum chi_a*a_phi` | `src/WENO.cu:1452-1457`, `src/WENO.cu:2187-2192`, `src/WENO.cu:2850-2855`. |

## x/y/z Direction Equivalence

The characteristic derivation is identical in all coordinate directions.  Only the memory layout and thread-block shapes differ.

| Direction | Flux kernel | Characteristic projection | Derivative |
| --- | --- | --- | --- |
| x | `compute_convective_term_weno_x`, `src/WENO.cu:831` | `src/WENO.cu:1090-1457` | `src/WENO.cu:1538-1558` |
| y | `compute_convective_term_weno_y`, `src/WENO.cu:1561` | `src/WENO.cu:1875-2192` | `src/WENO.cu:2196-2222` |
| z | `compute_convective_term_weno_z`, `src/WENO.cu:2225` | `src/WENO.cu:2538-2855` | `src/WENO.cu:2859-2885` |

## Formula-to-Code Map

| Mathematical object | Code variable | Code location |
| --- | --- | --- |
| $\rho,\rho u,\rho v,\rho w,\rho E,\rho\phi_a$ | `Q[0..n_var-1]` | `src/WENO.cu:1091-1127` and analogous y/z loading blocks |
| $U_k$ | `F[0]` | `src/WENO.cu:1113-1119` |
| momentum flux components | `F[1]`, `F[2]`, `F[3]` | `src/WENO.cu:1120-1122` |
| pressure | `F[4]` | `src/WENO.cu:1117-1124` |
| $c|\boldsymbol{k}|$ | `F[5]` | `src/WENO.cu:1129-1136` |
| Jacobian $J$ | `F[6]` | `src/WENO.cu:1137` |
| Roe velocity $\tilde{u},\tilde{v},\tilde{w}$ | `um`, `vm`, `wm` | `src/WENO.cu:1151-1153` |
| Roe scalar $\tilde{\phi}_a$ | `svm[l]` | `src/WENO.cu:1155-1158` |
| $\tilde{\gamma}_{tr}$, $c$, $\alpha_a$, $\chi_a$ | `gamma`, `cm`, `scalar_alpha`, `energy_coeff` | `src/WENO.cu:1166-1173`, `src/Thermo.cuh:49-110` |
| normal $\boldsymbol{n}$ | `kx`, `ky`, `kz` after normalization | `src/WENO.cu:1175-1180` |
| tangents $\boldsymbol{t},\boldsymbol{b}$ | `nx,ny,nz`, `qx,qy,qz` | `src/WENO.cu:1181-1199` |
| $\beta(Q)$, $\beta(F)$ | `sumBetaQ`, `sumBetaF` | `src/WENO.cu:1220-1230` |
| characteristic split $g^\pm$ | `vPlus`, `vMinus` | `src/WENO.cu:1279-1321`, `src/WENO.cu:1393-1435` |
| reconstructed characteristic flux | `fChar[l]` | WENO calls in `src/WENO.cu:1302`, `1321`, `1416`, `1435` |
| back-projected physical flux | `fc/gc/hc(...,l)` | `src/WENO.cu:1439-1457`, `2174-2192`, `2837-2855` |

## Two-Temperature Correctness Check

For the inviscid WENO characteristic path, the two-temperature extension is consistent with the derived frozen system:

| Requirement | Code | Result |
| --- | --- | --- |
| Add transported `rhoEve` scalar after species | `src/Parameter.cpp:647-655` | Correct. |
| Total energy closure uses $E=K+e_{tr}+E_{ve}$ | `src/InviscidScheme.cu:184-209`; primitive recovery uses the same thermodynamic helper | Correct. |
| Pressure differential includes $-\Gamma\,d(\rho E_{ve})$ | `src/Thermo.cuh:106-108` sets `alpha_Eve=-Gamma/c^2` | Correct. |
| Scalar right eigenvector for `rhoEve` has total-energy entry 1 | `src/Thermo.cuh:106-108` sets `energy_coeff[i_eve]=1` | Correct. |
| Characteristic scalar row uses $Q_{Eve}-\tilde{E}_{ve}Q_\rho$ | scalar-row loops in `src/WENO.cu` | Correct. |
| Back projection adds `rhoEve` scalar characteristic flux to total energy | `energy_coeff[l]*fChar[5+l]` in `src/WENO.cu:1452-1457`, y/z equivalents | Correct. |
| Convective `rhoEve` flux is $\rho E_{ve}U_k$ | scalar flux via `Q[5+l]*F[0]` | Correct. |

The same scalar coefficients are also used by the Roe Riemann solver path:

| Roe item | Code |
| --- | --- |
| Roe scalar average | `src/InviscidScheme.cu:654-657`. |
| `scalar_alpha` and `energy_coeff` | `src/InviscidScheme.cu:659-670`. |
| left projection includes `scalar_alpha*dq_scalar` | `src/InviscidScheme.cu:704-720`. |
| right projection includes `energy_coeff*b_scalar` | `src/InviscidScheme.cu:730-745`. |

## Relation to Viscous and Source Terms

This document derives only the inviscid characteristic WENO part.  The two-temperature governing equation also has source and viscous terms:

$$
\partial_t(\rho E_{ve})
+\nabla\cdot(\rho E_{ve}\boldsymbol{u})
=
\nabla\cdot\left[
\kappa_{ve}\nabla T_{ve}
+\sum_s(-J_s)E_{ve,s}(T_{ve})
\right]
+\theta_{tr:ve}
+\sum_s\dot{\omega}_sE_{ve,s}(T_{ve}).
$$

The inviscid WENO derivation above supplies only the convective flux

$$
\rho E_{ve}U_k.
$$

The viscous and source implementations are audited separately in the README and code comments:

| Term | Code |
| --- | --- |
| `rhoEve` viscous flux | `src/ViscousScheme.cu`. |
| nonequilibrium diffusion enthalpy $h_{tr}(T)+E_{ve}(T_{ve})$ | `src/Thermo.cuh:43-47`, `src/ViscousScheme.cu`. |
| chemical source $\sum_s\dot{\omega}_sE_{ve,s}(T_{ve})$ | `src/FiniteRateChem.cu`. |
| Landau-Teller update | `src/FieldOperation.cuh`, `src/RK.cuh`. |
