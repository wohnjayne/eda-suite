function g = sigmoid(z,T=1)
    g=1./(1+exp(-z/T));
end
