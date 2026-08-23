# Technical Brief: Multi-Physics Mathematical Mechanics & Reliability Models

This document details the governing equations, FEA continuum stress formulations, intermetallic growth kinetics, and solder joint fatigue models used in the **Micro_dashboard** optimization pipeline.

---

## 1. Continuum Thermal-Mechanical Mechanics

Thermal cycling induces differential strain due to thermal expansion coefficient mismatches across constituent materials ($\alpha_{\text{Si}} \approx 2.6\text{ ppm/K}$, $\alpha_{\text{Cu}} \approx 16.5\text{ ppm/K}$, $\alpha_{\text{FR4}} \approx 14\text{--}17\text{ ppm/K}$).

### Constitutive Stress-Strain Relationship
The total strain tensor $\boldsymbol{\epsilon}$ decomposes into elastic strain $\boldsymbol{\epsilon}_{\text{e}}$ and thermal strain $\boldsymbol{\epsilon}_{\text{th}}$:

$$\boldsymbol{\epsilon} = \boldsymbol{\epsilon}_{\text{e}} + \boldsymbol{\epsilon}_{\text{th}}$$

Where thermal strain vector for isotropic material behavior is:

$$\boldsymbol{\epsilon}_{\text{th}} = \alpha (T - T_{\text{ref}}) \mathbf{I}$$

Hooke's Law yields the stress tensor $\boldsymbol{\sigma}$:

$$\boldsymbol{\sigma} = \mathbf{C} : (\boldsymbol{\epsilon} - \alpha \Delta T \mathbf{I})$$

### Equivalent Von Mises Stress
Structural yield evaluation is governed by Von Mises equivalent stress $\sigma_{\text{vm}}$:

$$\sigma_{\text{vm}} = \sqrt{\frac{1}{2} \left[ (\sigma_{xx} - \sigma_{yy})^2 + (\sigma_{yy} - \sigma_{zz})^2 + (\sigma_{zz} - \sigma_{xx})^2 + 6(\tau_{xy}^2 + \tau_{yz}^2 + \tau_{zx})^2 \right]}$$

### Structural Margin of Safety (MoS)
Calculated relative to yield strength ($\sigma_{\text{yield}} = 220\text{ MPa}$):

$$\text{MoS} = \frac{\sigma_{\text{yield}}}{\sigma_{\max}} - 1$$

---

## 2. Solder Joint Thermal Fatigue Life Models

### Norris-Landzberg Modified Coffin-Manson Model
Accounts for temperature cycling amplitude, cycle frequency, and thermal peak conditions on solder joint fatigue:

$$\frac{N_{f,1}}{N_{f,2}} = \left( \frac{\Delta \gamma_2}{\Delta \gamma_1} \right)^m \left( \frac{f_1}{f_2} \right)^n \exp \left[ \frac{E_a}{k} \left( \frac{1}{T_{\max,1}} - \frac{1}{T_{\max,2}} \right) \right]$$

Where:
* $N_f$: Number of thermal cycles to failure.
* $\Delta \gamma$: Cyclic shear strain range.
* $f$: Cycle frequency.
* $T_{\max}$: Maximum junction temperature ($K$).
* $E_a / k$: Thermal activation parameter.

### Darveaux Crack Initiation & Propagation Model
Evaluates cumulative inelastic strain energy density ($\Delta W_{\text{ave}}$) per thermal cycle:

1. **Crack Initiation ($N_o$):**
   $$N_o = K_1 \cdot (\Delta W_{\text{ave}})^{K_2}$$

2. **Crack Propagation Rate ($da/dN$):**
   $$\frac{da}{dN} = K_3 \cdot (\Delta W_{\text{ave}})^{K_4}$$

3. **Total Fatigue Life ($N_f$):**
   $$N_f = N_o + \frac{a}{da/dN}$$

---

## 3. Intermetallic Compound (IMC) Kinetics

Intermetallic compound layer growth ($\text{Cu}_6\text{Sn}_5$ / $\text{Cu}_3\text{Sn}$) at the copper-solder interface follows thermally activated solid-state diffusion:

$$w(t) = w_0 + \sqrt{D(T) \cdot t}$$

Where $D(T)$ follows Arrhenius temperature dependence:

$$D(T) = D_0 \cdot \exp \left( -\frac{Q}{R \cdot T_{\text{op}}} \right)$$

* $w(t)$: Total IMC thickness after operating time $t$ (evaluated at 1 year).
* $Q$: Activation energy ($\text{J/mol}$).
* $R$: Universal gas constant ($8.314\text{ J/mol}\cdot\text{K}$).
* $T_{\text{op}}$: Mean operating temperature ($K$).