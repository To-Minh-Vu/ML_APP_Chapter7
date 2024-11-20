##### CHAPTER 7: FORWARD AND BACKWARD PROPAGATION #####

import numpy as np

#########   Define Add Gate ###########
class AddGate:
    def __init__(self):
        self.x = None
        self.y = None
    def Forward(self, x,y):
        self.x = x
        self.y = y
        return x+y 
    def Backward(self, d_out):
        return d_out, d_out

######## Define Multiply Gate ############
class MultiplyGate:
    def __init__(self):
        self.x = None
        self.y = None
    def Forward(self, x, y):
        self.x = x
        self.y = y
        return x*y
    def Backward(self, d_out):
        dx = d_out * self.y
        dy = d_out * self.x

        return dx, dy

######## Define Power Gate ################
class PowerGate:
    def __init__(self, power):
        self.x = None
        self.power = power
    def forward(self, x):
        self.x = x
        return x ** self.power
    def backward(self, d_out):
        return d_out * self.power * (self.x ** (self.power - 1))

if __name__ == "__main__":
    Multiply_gate1 = MultiplyGate()
    Multiply_gate2 = MultiplyGate()
    add_gate1 = AddGate()
    add_gate2 = AddGate()
    power_gate = PowerGate(2)

    ######## Init value ############
    w = 2
    x = -2
    b = 8
    y = 2
    
    ###### Forward propagation #########

    # Node 1: Compute c = w*x
    c = Multiply_gate1.Forward(w,x)

    #Node 2: Compute a = c + b
    a = add_gate1.Forward(c, b)
    
    #Node 3: Compute d = a - y
    d = add_gate2.Forward(a, -y)

    #Node 4: Compute e = d^2
    e = power_gate.forward(d)

    #Node 5: Compute J = 0.5 * e
    J = Multiply_gate2.Forward(0.5, e)

    print(f"Loss: {J}")

        ###### Backward propagation #########
    print("Backward Propagation Start")
    # Node 5: 
    _, A = Multiply_gate2.Backward(1)
    print("A= ",A)
    #Node 4:
    B = power_gate.backward(A)
    print("B = ", B)
    #Node 3:
    C, _= add_gate1.Backward(B)
    print("C = ", C)
    #Node 2: 
    D, E = add_gate1.Backward(B)
    print("D = ", D)
    print("E = ", E)
    #Node 5: 
    F,  _= Multiply_gate1.Backward(D)
    print("F = ", F)