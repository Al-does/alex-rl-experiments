# Mapping the MESS3 belief simplex to the Variant-2 two-state belief

## Reading guide and terminology

A distribution over three latent states has three coordinates constrained to
sum to one, so it has two intrinsic degrees of freedom. Its simplex
$\Delta^2$ is therefore a filled triangle. A distribution over two latent
states has one degree of freedom, so $\Delta^1$ is a line segment. Thus
“three-state to two-state” and “two-dimensional to one-dimensional” describe
the same reduction here.

There are two related but different operations in this note:

1. **Aggregate a belief point:** add the probabilities of states 0 and 1.
   This is an instantaneous linear map from the triangle to the line.
2. **Filter coarsened observations:** rerun Bayesian filtering after hiding
   the distinction between tokens 0 and 1. This is a map between histories,
   not just between current belief points.

The first operation produces $s_t$; the second produces $c_t$. Keeping these
operations separate resolves most of the apparent paradoxes below.

## 1. State aggregation

Let the fine latent state be

$$
X_t\in\{0,1,2\},
$$

and define the macrostate

$$
Z_t=f(X_t),\qquad
f(0)=f(1)=A,\quad f(2)=B.
$$

For a fine belief

$$
b=(b_0,b_1,b_2)\in\Delta^2,
$$

the natural state-aggregation map is

$$
\pi(b)=q=(q_A,q_B)=(b_0+b_1,b_2)\in\Delta^1.
$$

Using row-vector beliefs, this is the linear map

$$
q=bC,\qquad
C=
\begin{pmatrix}
1&0\\
1&0\\
0&1
\end{pmatrix}.
$$

Thus the scalar coordinate of the two-state belief can be taken to be

$$
s=q_B=b_2,
$$

since $q_A=1-s$.

## 2. Geometry of the simplex map

The three-state belief space $\Delta^2$ is a triangle, while the two-state
belief space $\Delta^1$ is a line segment.

Under $\pi$:

- vertex $e_0=(1,0,0)$ maps to the $A$ endpoint;
- vertex $e_1=(0,1,0)$ maps to the same $A$ endpoint;
- vertex $e_2=(0,0,1)$ maps to the $B$ endpoint;
- every line of constant $b_2=s$ collapses to the single point
  $(1-s,s)$.

The fiber above a coarse belief $q=(q_A,q_B)$ is

$$
\pi^{-1}(q)
=
\left\{
(u,q_A-u,q_B):0\leq u\leq q_A
\right\}.
$$

So the discarded direction is exactly redistribution of probability between
states 0 and 1.

For example, all of

$$
(0.70,0.10,0.20),\qquad
(0.40,0.40,0.20),\qquad
(0.05,0.75,0.20)
$$

map to the same coarse belief $(0.80,0.20)$. Geometrically, they lie on the
same line segment parallel to the edge joining $e_0$ and $e_1$. The map does
not choose one of these points as the “true” inverse; it declares their
within-lump differences irrelevant.

A useful symmetry-adapted coordinate system is

$$
s=b_2,\qquad a=b_0-b_1.
$$

The inverse reconstruction is

$$
b_0=\frac{1-s+a}{2},\qquad
b_1=\frac{1-s-a}{2},\qquad
b_2=s,
$$

with

$$
0\leq s\leq 1,\qquad |a|\leq 1-s.
$$

Equivalently,

$$
b=
\underbrace{\left(\frac{1-s}{2},\frac{1-s}{2},s\right)}_{\text{symmetric/coarse component}}
+
\underbrace{\frac{a}{2}(1,-1,0)}_{\text{antisymmetric component}}.
$$

The three-to-two map keeps $s$ and drops $a$.

There is also a canonical lift of a two-state belief back into the triangle:

$$
\iota(q_A,q_B)=
\left(\frac{q_A}{2},\frac{q_A}{2},q_B\right).
$$

The composition

$$
S=\iota\circ\pi,\qquad
S(b)=
\left(\frac{b_0+b_1}{2},\frac{b_0+b_1}{2},b_2\right),
$$

is the orthogonal symmetrization of the fine belief onto the line $b_0=b_1$.
It is idempotent: $S^2=S$.

In linear-algebra language, the lost direction is the kernel direction
$(1,-1,0)$. Within the simplex, quotienting by that entire direction lowers
the intrinsic dimension from two to one. This is why a picture of the map
should show whole parallel line segments collapsing, rather than merely
folding the triangle in half.

One subtlety: quotienting only by the discrete label swap $0\leftrightarrow1$
would retain $\lvert b_0-b_1\rvert$ and would still be two-dimensional.
The one-dimensional map above is stronger: it deliberately discards all
within-lump information.

## 3. Why this aggregation is dynamically valid in Variant 2

For each action $u$, let $T_u$ be the $3\times3$ controlled transition
matrix. Strong lumpability with respect to $\{A=\{0,1\},B=\{2\}\}$ means

$$
\sum_{j\in A}T_u(0,j)=\sum_{j\in A}T_u(1,j)
$$

and

$$
T_u(0,2)=T_u(1,2).
$$

Equivalently, there is a $2\times2$ matrix $\bar T_u$ satisfying

$$
T_uC=C\bar T_u.
$$

This holds for every action in Variants 1 and 2, but fails in Variant 3.
Consequently,

$$
(bT_u)C=(bC)\bar T_u:
$$

prediction commutes exactly with aggregation. Reward also factors through the
same map because reward depends only on whether the state is 2:

$$
r(x,u)=\bar r(f(x),u).
$$

For Variant 2, the reduced transition matrices are approximately

$$
\bar T_{\mathrm{noop}}=
\begin{pmatrix}
0.900000&0.100000\\
0.600000&0.400000
\end{pmatrix},
$$

$$
\bar T_{\mathrm{positive}}=
\begin{pmatrix}
0.667572&0.332428\\
0.870509&0.129491
\end{pmatrix},
$$

$$
\bar T_{\mathrm{negative}}=
\begin{pmatrix}
0.975808&0.024192\\
0.870509&0.129491
\end{pmatrix}.
$$

Thus the reward-relevant controlled dynamics genuinely close on the scalar
probability of macrostate $B$.

This closure gives the coarse state a precise sufficiency claim: before the
next observation arrives, $q_t=b_tC$ contains everything needed to predict
the next macrostate distribution and the expected reward for each action.
It does **not** say that $q_t$ contains everything needed to process the
identity of the next fine token. Transition prediction commutes with
aggregation, but fine-token Bayesian updating generally does not.

## 4. Why the separately computed coarse filter is not just $b_2$

The original three-symbol emission matrix does **not** factor through the
state aggregation: states 0 and 1 have different probabilities of emitting
tokens 0 and 1. Therefore an exact update after observing the identity of a
fine token generally depends on the discarded coordinate $a=b_0-b_1$.

For intuition, token 0 is evidence favoring latent state 0 over state 1,
whereas token 1 supplies the opposite evidence. Because states 0 and 1 are
sticky, that evidence affects predictions several steps later. Even when the
control problem treats those states identically *at the current step*, the
exact fine observer can use their identities to sharpen its future estimate
of state 2.

To obtain a self-contained two-state HMM, also coarsen observations:

$$
g(0)=g(1)=N\quad\text{(``not 2'')},\qquad g(2)=2.
$$

The reduced emission matrix is then

$$
\bar E=
\begin{pmatrix}
P(N\mid A)&P(2\mid A)\\
P(N\mid B)&P(2\mid B)
\end{pmatrix}
=
\begin{pmatrix}
0.925&0.075\\
0.150&0.850
\end{pmatrix}.
$$

If $c_{t-1}=P(Z_{t-1}=B)$, first predict

$$
p_t^-=(1-c_{t-1})\bar T_u(A,B)
      +c_{t-1}\bar T_u(B,B).
$$

Then update with the coarsened observation:

$$
c_t=
\begin{cases}
\displaystyle
\frac{0.85p_t^-}{0.075(1-p_t^-)+0.85p_t^-},
&g(Y_t)=2,\\[1.2em]
\displaystyle
\frac{0.15p_t^-}{0.925(1-p_t^-)+0.15p_t^-},
&g(Y_t)=N.
\end{cases}
$$

This $c_t$ is the “coarse target” used by the separate two-state filter.

It is important to distinguish:

1. **Projected full belief**
   $$
   s_t=\pi(b_t)_B=b_{t,2},
   $$
   where $b_t$ conditions on the complete token identities.

2. **Coarse-filter belief**
   $$
   c_t=P(Z_t=B\mid g(Y_{1:t}),u_{1:t-1}),
   $$
   where the filter never receives the distinction between tokens 0 and 1.

In general,

$$
c_t\neq s_t.
$$

There is therefore no general pointwise function that takes one realized
fine posterior $b_t$ and recovers the separately evolved $c_t$. The map
from a fine belief point to its aggregated occupancy is $\pi(b_t)$; the map
from the fine filtering process to the coarse filtering process acts on
information histories.

Another way to say this is that $b_t$ is a sufficient statistic for an
observer who will continue receiving fine observations, but it was not
designed to remember which coarsened-history equivalence class produced it.
The coarse filter is its own recursively maintained statistic. One can obtain
$c_t$ by retaining the coarsened history or by updating that statistic online;
one should not expect to reconstruct it afterward from the current $b_t$
alone.

If actions are treated as fixed known controls and
$\mathcal G_t\subset\mathcal F_t$ are respectively the coarse and fine
observation histories, the two targets satisfy the useful tower-property
relationship

$$
c_t
=
\mathbb E[s_t\mid\mathcal G_t].
$$

So $c_t$ is the conditional average of the projected full posterior over
all fine histories that become indistinguishable after replacing tokens 0
and 1 by $N$.

The tower property also gives an exact information-loss identity:

$$
\mathbb E[(s_t-c_t)^2]
=
\mathbb E[\operatorname{Var}(s_t\mid\mathcal G_t)].
$$

Thus the mean squared gap between $s_t$ and $c_t$ is the fine-history
variation removed by observation coarsening. By the law of total variance,

$$
\operatorname{Var}(s_t)
=
\operatorname{Var}(c_t)
+
\mathbb E[\operatorname{Var}(s_t\mid\mathcal G_t)],
$$

so $c_t$ is necessarily no more variable than $s_t$. It is a conditional
average, not a noisier approximation to the same random variable.

The conditioning statement needs one control-related qualification. It is
exact when the action sequence is fixed, conditioned upon, or generated from
the coarse history. If actions depend on fine token identities, the actions
themselves can reveal fine information. Analyses should therefore condition
on identical controls, as the paired frozen-policy replay does, or explicitly
include actions in both filtrations.

## 5. Interpreting the MSE difference

If a representation retains only the coarse coordinate, the best canonical
fine-belief reconstruction is

$$
\tilde b_t=
\left(\frac{1-c_t}{2},\frac{1-c_t}{2},c_t\right).
$$

This is the symmetry-preserving lift. It is also the conditional-mean lift
when the data distribution is invariant under exchanging states 0 and 1.
Without that symmetry, a learned decoder could choose a different location
along each fiber, so the lift should be understood as a principled reference
rather than an arbitrary inverse of $\pi$.

For simplex-constrained predictions, its coordinate-averaged squared error
against the full belief is

$$
\frac{1}{3}\|\tilde b_t-b_t\|_2^2
=
\frac{1}{2}(c_t-s_t)^2+\frac{1}{6}a_t^2.
$$

The full-belief error therefore contains two components that the coarse-target
error does not:

1. the omitted antisymmetric information $a_t=b_{t,0}-b_{t,1}$;
2. the difference between the full-information reward-state posterior $s_t$
   and the coarse-history posterior $c_t$.

This makes a much lower MSE to $c_t$ exactly what one would expect if the
agent implements the reduced filter. Raw MSEs for a three-vector and a scalar
should not be compared without accounting for dimensionality and target
variance; normalized MSEs, symmetry-coordinate probes, and initialization
floors are more informative.

Because $c_t$ and $s_t$ are highly correlated, the strongest mechanistic
test would probe their unique residuals:

$$
\tilde c_t=c_t-\mathbb E[c_t\mid s_t],
\qquad
\tilde s_t=s_t-\mathbb E[s_t\mid c_t].
$$

Preferential decodability of $\tilde c_t$ would be stronger evidence that
the representation implements the coarse filter rather than merely encoding
the shared reward-state component.

Probe fit alone still establishes linear accessibility, not causal policy
use. Three questions should be kept distinct:

1. **Accessibility:** can a probe decode $s_t$, $c_t$, or the discarded
   coordinate $a_t$?
2. **Computation:** does the decoded variable obey the corresponding Bayes
   recursion across time?
3. **Use:** does changing or ablating that variable alter action logits and
   behavior?

## 6. Why independent token flips distinguish the targets

A global exchange of every token 0 with token 1 is not discriminating. It is
an exact symmetry of the fine HMM:

$$
(b_{t,0},b_{t,1},b_{t,2})
\longmapsto
(b_{t,1},b_{t,0},b_{t,2}).
$$

It flips the sign of $a_t$ but leaves both $s_t=b_{t,2}$ and $c_t$ unchanged.
Consequently, both a fine-$s_t$ policy and a coarse-$c_t$ policy can be
invariant under a global swap.

Independently flipping each non-2 token with probability $1/2$ is different.
It preserves the complete coarse observation sequence—the positions of token
2 versus “not 2”—and therefore preserves $c_t$ exactly, while changing the
fine token pattern and generally changing $s_t$. Paired histories produced
this way trace different fine-filter paths inside the same coarse-history
equivalence class.

This yields a direct diagnostic:

- invariance of hidden states, decoded beliefs, and action logits supports a
  coarse-filter-like computation;
- systematic changes that track the recomputed $s_t$ support retained
  fine-history information;
- hidden-state changes without policy changes support a mixed representation
  whose fine component is not causally used by the action head.

## 7. Visualization consequences

The geometry suggests several complementary figures:

1. **Triangle-to-line quotient:** draw the MESS3 triangle with constant-$b_2$
   fibers parallel to the $e_0$--$e_1$ edge, then connect each fiber to its
   point on a neighboring two-state line.
2. **Symmetry coordinates:** redraw the triangle in $(s,a)$ coordinates.
   Vertical position can represent $s=b_2$, horizontal position
   $a=b_0-b_1$, and the coarse map becomes horizontal collapse onto $a=0$.
3. **Information comparison:** scatter $s_t$ against $c_t$. The conditional
   vertical spread at a fixed $c_t$ visualizes fine information removed by
   coarsening; its expected squared width is
   $\mathbb E[(s_t-c_t)^2]$.
4. **Paired trajectories:** show factual and independently randomized
   histories with identical $c_t$ trajectories but diverging $s_t$
   trajectories, then compare their decoded states and policy outputs.

No single plot should conflate the geometric projection $\pi(b_t)$ with the
history-level coarse filter. A triangle colored only by $b_2$ illustrates the
former; comparing paired filtering trajectories is needed to illustrate the
latter.

## 8. Compact conceptual summary

The triangle-to-line map is straightforward:

$$
(b_0,b_1,b_2)\mapsto(b_0+b_1,b_2).
$$

What requires care is deciding which posterior lives on that line.
$s_t=b_{t,2}$ projects a posterior formed with the full token history.
$c_t$ is formed by a different observer that never saw the 0/1 distinction.
They concern the same macrostate and share the same simplex coordinate system,
but they condition on different information. Geometry explains what is
discarded at one instant; filtering explains how discarded information
changes the posterior over time.
