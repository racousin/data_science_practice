# Parameter Initialization

Before the first step, every weight in the network already has a value, chosen
by a line of code most people never read. That value has two jobs. It must
**differ between units**, or they stay copies of each other forever. And it must
have **the right scale**, or the signal dies or explodes before it reaches the
loss. Get either wrong and no optimizer, schedule or regularizer later in this
session can recover it.


<!-- notes: 25 minutes. Two demos carry the lesson: the constant-init network
whose 256 units stay identical (print the rank), and the first loss of a
N(0, 1) network against ln 10. Derive the variance argument on the board in
three lines; do not derive Xavier's harmonic mean. The special-cases slide is
reference: point at it, do not read it. -->

---

## Job one: break the symmetry

If two hidden units start with the same incoming weights, they compute the same
output on every input, receive the same gradient, and take the same step. After
any number of steps they are still identical: the layer has the width of one
unit, however many you declared.

```python
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.01)     # every unit the same
        nn.init.zeros_(m.bias)
```


---

## Job two: set the scale

One unit of a `Linear` layer sums $n_{in}$ products. With zero-mean weights
drawn independently of zero-mean inputs,

$$
\operatorname{Var}(y) = n_{in}\,\operatorname{Var}(w)\,\operatorname{Var}(x)
$$

so each layer multiplies the variance of the signal by $n_{in}\operatorname{Var}(w)$.
Stack thirty layers and that factor is raised to the thirtieth power: it has to
be 1, or the signal is gone. Setting it to 1 gives the three classic rules:

$$
\operatorname{Var}(w) = \frac{1}{n_{in}}\ \text{(LeCun)} \qquad
\frac{2}{n_{in} + n_{out}}\ \text{(Xavier)} \qquad
\frac{2}{n_{in}}\ \text{(He)}
$$


![Weight distributions of a 784-512-256-128-64-32 network under a fixed N(0, 0.1²), Xavier and He](assets/nn/weight_distributions.png)
