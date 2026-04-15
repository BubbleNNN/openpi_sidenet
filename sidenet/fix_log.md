2026.4.2
- 问题1：ft输入没有办法被normalize
    这个问题来自两部分：
    1. data transform的时候把ft命名成了force_torque, 但是之前计算好的norm stats文件中ft的名称是force_sensor
        注意，这里在训练的时候行为是，从底层dataset里面读键名为observation.ft_sensor_window的value，如果有，那么force_sensor就是它，反之就是单帧的ft
        把单帧ft包裹成多帧的逻辑交给了dataset wrapper
    2. ft window一共有6帧，所以在normalize的时候需要特殊处理。现在6帧的构造是在创建dataset的时候完成的，这个过程在normalize之前，所以没办法先normalize然后再构造
    解决方案：
    1. 在config.py line327新加一个辅助方法，把norm stats文件中的force_sensor改成force_torque
    2. 在config.py line363用更新后的norm stats替换原有的norm stats
    3. 维度的问题，由于整个数据集一起计算得到同样的norm states，所以不同fram的ft在normalize的时候做的运算是相同的。所以只要norm stats和input data的key相同，广播就会自动handle多frame ft的normalization
    已进行测试，问题解决
- 问题2：训练的时候存储ckpt会额外存储norm stats
    解决方案：train.py line480,不再存储norm stats
- 问题3： VLM backbone injection 关闭
    解决方案：在TrainConfig中关闭injection
临走前挂上的后台进程：1245955
2026.4.3
- 需要合并代码，把总部的代码和自己的代码合并，并且打包成docker
    未同步部分：1. pi0_pytorch.py的各种模型变体 2. policy.config的加载逻辑和最后的create_trained_policy_from_model函数
    3. rb1 policy 和rb1 xhand policy两个datapolicy 4. download.py 5.utils的最后一个函数
- 问一下维度问题
    总部使用的是灵巧手。需要把代码合并成总部一致的代码
- 问题1: 输入ft数据和模型的dtype不一致
    解决方案：在CNNencoder forward里面加入数据转换的代码
- 问题2:loss巨大
    尝试1:禁用所有来自sidenet的injection，发现训练的loss还是飞到天上，说明至少主要原因不在sidenet