# Instance-Segment-BASI


# train 1

0. pool size = 50

1. pos_weight=5 训练segment 100000

2. pos_weight=3 训练segment 350000

3. add attention class 0.1 * loss_classes 100000

4. add attention class 0.2 * loss_classes 200000

5. 同时训练 attention class 和 segment 230000

6. result in class,segment,together


# train 2

0. pool size = 90

1. pos_weight = 3 + attention class 0.2 * loss_classes 同时训练 attention class 和 segment 310000

2. result in begin/first


# train 3

0. pool size = 90

1. segment 3 类

2. result in begin/second


# train 4

0. segment border

1. segment 4 类

2. result in begin/third


