# Grid-World

### Let the Robot find a way in the Grid World!
```
Avoid (1, 3) and heading to (0, 3). Plus, (1, 1) is wall!
init P={0.1, 0.1, 0.8} for robot to turn left or right or go forward when it gets the order to go straight!
fun thing is: recommend route changes when P comes to {0.1, 0.5, 0.85} & {0.1, 0.1, 0.89}
check!
 ```
* when P={0.1, 0.1, 0.8}, recommed route is: \
![2](https://raw.githubusercontent.com/SJHNJU/Grid-World/master/routes/route2.png)

* but when P={0.1, 0.05, 0.85}, route change to: \
![1](https://raw.githubusercontent.com/SJHNJU/Grid-World/master/routes/route1.png)

* and if P={0.1, 0.01, 0.89}, route is: \
![3](https://raw.githubusercontent.com/SJHNJU/Grid-World/master/routes/route3.png)

**Check! it is fun when it comes to challenge how we look at the things**


### method:
- Value iteration in reinforcement learning
```
Repeat:
expectation(a) = sum(Ps,a(s_) * V(s_) for s_ in S_) ----S_ is the probable next state when taking action a in state s
V(S) := R(S) + max( gamma * expectation(a) for a in A) --- A is the available actions in state S
and:
V(S) will be close to V*(S) --V(S) -> V*(S)

find best policy:
optimal policy equation = argmax(expectation(a) for a in A)
```
- Q-learning
```
target:generate a state-action-value table
Q(s,a) := (1- alpha)Q(s,a) + alpha[R(s,a) + gamma * maxQ(s_,a)]
```
