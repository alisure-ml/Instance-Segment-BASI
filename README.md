# Instance-Segment-BASI


# train

1. pos_weight=5 训练segment 100000

2. pos_weight=3 训练segment 350000

3. add attention class 0.1 * loss_classes 100000

4. add attention class 0.2 * loss_classes 200000

5. 同时训练 attention class 和 segment 230000

