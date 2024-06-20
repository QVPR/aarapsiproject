# VPR Verification Statistical Features

## Shared Notation Definitions
- For input vector $x$ with dimension $n$.
- $\hat{x}$ represents the sorted form of $x$, in ascending order.
- All indices start from 1 for the first index, and are denoted by a subscript i.e. $x_i$ is the $i^{th}$ element of $x$.
- $\bar{x}$ represents the subset of $x$ such that every element is a local minima, i.e. fulfilling the criteria of $x\_{i-1} \gt x\_{i} \lt x\_{i+1}$, with dimension $\bar{n}$.
- $\bar{X}$ represents the indices for each element in $\bar{x}$ as they correspond to $x$ such that $0 \le \bar{X}_{i} \le n$, with dimension $\bar{n}$.
- Set notation $[a,b]$ indicates inclusive of $a$ and $b$ and all integers between $a$ and $b$.
- When an operation leads to division by zero, infinity, or undefined, we substitute the result with zero.

## Statistical Feature Definitions
For each VPR vector, the following statistical features are extracted and appended to a vector at the index corresponding to their list enumeration, for a total of 48 statistical features per vector:

Index 1 - <b>Normalized Mean</b>:

We use the mean, normalized by the range $R$:
$$\Huge R = \max{(x)} - \min{(x)} $$
$$\Huge \mu = \frac{1}{n} \sum_{i=1}^{n} x_i $$
$$\Huge C_1 = \frac{1}{R} \cdot \mu $$  

Index 2 - <b>Normalized Standard Deviation</b>:

We use the standard deviation, normalized by the range $R$; when $R = 0$, $C_2 = 0$:
$$\Huge \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2} $$
$$\Huge C_2 = \frac{1}{R} \cdot \sigma $$  

Index [3-23] - <b>Normalized Percentile Thresholds</b>:

We calculate each of these elements as percentile thresholds, normalized by the range $R$, as values $\min{(x)} \le P \le \max{(x)}$:

$$\Huge C_{i+3} = \frac{1}{R} \cdot \hat{x}_{\text{round}\left\(0.05 \cdot i \cdot n \right\)} \text{  for } i \in [0,20]$$
  
Index [24-28] - <b>Normalized Smallest Values</b>:

We set these elements to $s_{[1,5]}$, where $s_i$ is the $i^{th}$ smallest value in the vector $x$, normalized by the range $R$:
$$\Huge C_{[24,28]} = \frac{1}{R} \cdot \set{s_1, s_2, s_3, s_4, s_5} $$
  
Index [29-33] - <b>Normalized Largest Values</b>:

We set these elements to $b_{[1,5]}$, where $b_i$ is the $i^{th}$ biggest value in the vector $x$, normalized by the range $R$:
$$\Huge C_{[29,33]} = \frac{1}{R} \cdot \set{b_1, b_2, b_3, b_4, b_5} $$
  
Index 34 - <b>Normalized Sum</b>:

$$\Huge C_{34} = \frac{1}{n \cdot R} \cdot \sum_{i=1}^{n} x_i $$  

Index 35 - <b>Normalized Range</b>:

$$\Huge C_{35} = 1 - \frac{\min{x}}{\max{x}} $$  

Index 36 - <b>Normalized Inter-Quartile Range (IQR)</b>:

Using $P_{75}$ and $P_{25}$ as the values for the $75^{th}$ and $25^{th}$ percentile thresholds:
$$\Huge C_{36} = \frac{1}{R} \cdot (P_{75} - P_{25}) $$  
 
  
Index 37 - <b>Normalized Mean/Median Skew</b>:

$$\Huge C_{37} = \frac{C_{1}}{P_{50}} $$  

Index 38 - <b>Normalized IQR Skew</b>:

Where $P_{75}$, $P_{50}$, and $P_{25}$ are the values for the $75^{th}$, $50^{th}$, and $25^{th}$ percentile thresholds:
$$\Huge \frac{P_{75}-P_{50}}{P_{75}-P_{25}} $$  
  
Index 39 - <b>Normalized Standard Deviation of Minima Values</b>:

The standard deviation of minima values, $\bar{x}$, normalised by the dimension of $x$:
$$\Huge \bar{\mu} = \frac{1}{\bar{n}} \sum\_{i=1}^{\bar{n}} \bar{x}\_i $$
$$\Huge \bar{\sigma} = \sqrt{\frac{1}{\bar{n}} \sum\_{i=1}^{\bar{n}} (\bar{x}\_i - \bar{\mu})^2} $$
$$\Huge C_{39} = \frac{1}{n} \cdot \bar{\sigma} $$

Index 40 - <b>Normalized Standard Deviation of Minima Indices</b>:

The standard deviation of minima indices, $\bar{x}$, normalised by the range of $x$:
$$\Huge \bar{M} = \frac{1}{\bar{n}} \sum\_{i=1}^{\bar{n}} \bar{X}\_i $$
$$\Huge \bar{\Sigma} = \sqrt{\frac{1}{\bar{n}} \sum\_{i=1}^{\bar{n}} (\bar{X}\_i - \bar{M})^2} $$
$$\Huge C_{40} = \frac{1}{R} \cdot \bar{\Sigma} $$

Index 41 - <b>Normalized Minima Sensitivity</b>:

Where $s_i$ is the $i^{th}$ smallest value in the vector $x$. Result is normalized by the range $R$:
$$\Huge C_{41} = \frac{1}{R} \cdot (s\_2 - s\_1) $$

Index 42 - <b>Normalized Maxima Sensitivity</b>:

Where $b_i$ is the $i^{th}$ smallest value in the vector $x$. Result is normalized by the range $R$:
$$\Huge C_{42} = \frac{1}{R} \cdot (b\_2 - b\_1) $$
  
Index 43 - <b>Normalized Minima Separation Sum</b>:
$$\Huge V_i = \frac{1}{n \cdot R} \cdot \bigg\(\bar{x}\_{i} - \min{(x)}\bigg\) \cdot \bigg\(\bar{X}\_{i} - \text{argmin}(x)\bigg\) \text{  for } i \in [0, \bar{n}] $$
$$\Huge C_{43} = \frac{1}{\bar{n}} \cdot \sum_{i=1}^{n} V_i $$  

Index 44 - <b>Normalized Minima Separation Range</b>:
$$\Huge C_{44} = \max{V} - \min{V} $$ 
  
Index 45 - <b>Normalized Median of Bounded Minima Region</b>:  
$$\Huge k = \text{argmin}(x) $$
$$\Huge \alpha = \max{(\set{0,k-10})} $$
$$\Huge \beta = \min{(\set{k+10,n})} $$
$$\Huge \epsilon = \text{median}(x_{[\alpha,\beta]}) $$
$$\Huge C\_{45} = \frac{1}{R} \cdot \left\lvert \epsilon - \min{(x)} \right\rvert $$

Index 46 - <b>Percentage Under Median in Bounded Minima Region</b>:

$$
\Huge
E\_i = \begin{cases}
                 1 & x\_i \lt \epsilon \\
                 0 & x\_i \gt \epsilon
\end{cases}
$$

$$\Huge C_{46} = \frac{1}{n} \cdot \sum_{i=1}^{n} E_i $$  

Index 47 - <b>Minima Range Ratio [1]</b>:

$$\Huge C\_{47} = \frac{(s\_{2} - s\_{1})}{(s\_{n}-s_{1})} $$

Index 48 - <b>Average Minima Gradient [1]</b>:

$$\Huge k = \text{argmin}(x) $$

$$
\Huge
C\_{48} = \begin{cases}
                 \frac{1}{2} \cdot (x\_{k+1} + x\_{k-1}) - x\_{k} & 1 < k < n \\
                 x\_{k+1} - x\_{k} & k = 1 \\
                 x\_{k-1} - x\_{k} & k = n
\end{cases}
$$

[1] H. Carson, J. J. Ford, and M. Milford, “Predicting to improve: Integrity measures for assessing visual localization performance,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 9627–9634, 2022.
