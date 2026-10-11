---
title: 'Online softmax: why FlashAttention reads each row once'
date: 2026-10-11
description: Folding softmax's three passes into one is what lets FlashAttention compute attention tile by tile.
layer: kernel
draft: true
---

> Sample post that shows off code highlighting and math. `draft: true` keeps it out of the build: it only shows up under `npm run dev`. Rewrite it or delete it.

## Safe softmax takes three passes

To avoid overflow, softmax subtracts the row maximum first:

$$
m = \max_j x_j,\qquad d = \sum_j e^{x_j - m},\qquad y_i = \frac{e^{x_i - m}}{d}
$$

One pass for $m$, one for $d$, one to write $y$. Each pass reads the whole row from HBM, so a memory-bound kernel pays for three times the traffic.

## Correct as you go

$d$ can be accumulated before $m$ is known. Whenever the running maximum grows, rescale what has been summed so far:

$$
m_j = \max(m_{j-1}, x_j),\qquad d_j = d_{j-1}\, e^{m_{j-1} - m_j} + e^{x_j - m_j}
$$

Finding $m$ and $d$ is now a single pass:

```python
import math

def online_softmax(xs):
    m, d = float("-inf"), 0.0
    for x in xs:
        m_new = max(m, x)
        d = d * math.exp(m - m_new) + math.exp(x - m_new)
        m = m_new
    return [math.exp(x - m) / d for x in xs]
```

## Inside attention

In attention, softmax is immediately followed by a multiply with $V$. The output accumulator $o$ can be rescaled by the same factor, which removes the pass that writes $y$:

$$
o_j = o_{j-1}\, e^{m_{j-1} - m_j} + e^{x_j - m_j}\, v_j,\qquad \text{out} = \frac{o_N}{d_N}
$$

FlashAttention runs this recurrence per tile rather than per element: load a block of $K$ and $V$ into shared memory, take the block's local maximum, rescale $m$, $d$ and $o$, move on. The $N \times N$ score matrix never goes back to HBM.
