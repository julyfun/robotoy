## just want code

- splitted tests!
- cmake manually

## bugs

- 在最高 v 前进段采用积分优化估计的 remaining_t 来估计 acc_suggested，但是这个 t 显然是偏大的，实际上到达目标 v 的时候目标 v 和 acc 都会下降，导致会追不上（即使有调整系数），速度始终偏大. 该 bug 在 am ** 2 > vm * jm 的情况下尤为明显。
    - 但是这么大的 am 很少见.
    - 可考虑更好地估计 remaining_t
