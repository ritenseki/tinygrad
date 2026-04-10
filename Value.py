import math

class Value:
    def __init__(self, data, _children=()):
        self.data = data;
        self.grad = 0.0;
        self._prev = set(_children)
        self._backward = lambda: None
    
    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    
    # Basic functions

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other))
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += (self.data ** other.data) * math.log(self.data) * out.grad

        out._backward = _backward
        return out


    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __rtruediv__(self, other):
        return Value(other) / self

    # These func combined by all func above;

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** (-1))

    def tanh(self):
        tanh_value = math.tanh(self.data)
        out = Value(tanh_value, (self,))
        def _backward():
            self.grad += (1 - tanh_value ** 2) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        data = self.data if self.data > 0 else 0
        grad = 1 if self.data > 0 else 0
        out = Value(data, (self,))
        def _backward():
            self.grad += grad * out.grad
        
        out._backward = _backward
        return out
            

    # Backward  function

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()