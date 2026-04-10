# Tiny Autograd 教学记录

## 项目目标

从零实现一个最小的自动微分引擎，最终能训练一个简单的神经网络。

---

## 第一阶段：Value 核心

### Value 类的结构

每个 `Value` 对象代表计算图中的一个节点，包含：

- `data`：这个节点的值
- `grad`：这个节点对最终输出的梯度（初始为 0）
- `_prev`：生成这个节点的子节点（用于构建计算图）
- `_backward`：如何把梯度传给子节点的函数

---

### 反向传播的核心直觉

**链式法则的直觉理解：**

向后传播的梯度 = 上层传来的梯度 × 本层的局部偏导

例如：

```
a = Value(2.0)
b = Value(3.0)
c = a * b    # c = 6.0
d = c * 4.0  # d = 24.0
```

想知道 `dd/da`：
- `dd/dc = 4`（上层传来的梯度，即 `out.grad`）
- `dc/da = b = 3`（本层对 a 的局部偏导）
- `dd/da = 4 * 3 = 12`

所以 `_backward` 里写的是：
```python
self.grad += other.data * out.grad
#            本层局部偏导  上层梯度
```

**为什么用 `+=` 而不是 `=`：**

一个节点可能在计算图中被多次使用，例如 `a + a`。
每条路径都会传来一份梯度，需要全部累加，而不是覆盖。

---

### 已实现的运算

| 运算 | 方法 | 局部偏导 |
|------|------|----------|
| 加法 | `__add__` | 对两个输入都是 1 |
| 乘法 | `__mul__` | 对 self 是 other，对 other 是 self |
| 幂运算 | `__pow__` | `n * x**(n-1)` |
| 取负 | `__neg__` | `self * (-1)` 组合实现 |
| 减法 | `__sub__` | `self + (-other)` 组合实现 |
| 除法 | `__truediv__` | `self * (other ** -1)` 组合实现 |
| tanh | `tanh` | `1 - tanh(x)**2` |
| ReLU | `relu` | x>0 时为 1，否则为 0 |

组合运算（`__neg__`、`__sub__`、`__truediv__`）不需要自己写 `_backward`，
因为它们复用了已有运算，梯度会自动通过那些运算传播。

`__radd__`、`__rmul__` 等是为了处理 `2 + Value(3)` 这类顺序反过来的情况。
Python 在 `int.__add__(Value)` 失败时会自动尝试 `Value.__radd__(int)`。
实现上直接复用正向运算即可，不需要额外的 `_backward`。

---

### backward() 的工作方式

```python
def backward(self):
    # 1. 拓扑排序：保证每个节点在其所有子节点之后处理
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # 2. 从输出节点开始，梯度为 1
    self.grad = 1.0

    # 3. 从后往前，依次调用每个节点的 _backward
    for node in reversed(topo):
        node._backward()
```

反转拓扑序 = 从输出往输入传播，保证每个节点在被用到之前已经收到了完整的梯度。

---

---

## 项目完成

autograd 引擎核心功能已全部实现。
下一步可以尝试 karpathy/nanoGPT，从零实现 GPT。
