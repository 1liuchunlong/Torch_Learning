import matplotlib.pyplot as plt
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x * w

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y - y_pred)**2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

cost_list = []
epoch_list = []
lr = 0.01
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= lr * grad_val
    cost_list.append(cost_val)
    epoch_list.append(epoch + 1)
    print(f'epoch {epoch}: w = {w} cost = {cost_val}')

plt.figure(figsize=(10, 8))
plt.plot(epoch_list, cost_list, marker = 'o', linestyle = '-', linewidth = 2)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('After training 100 epochs')
plt.grid(True)
plt.show()


print(f'Predict x = 4.0 y = {forward(4)}')