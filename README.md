# GA for CVRPTW

使用遗传算法解决带时间窗口与载重约束的车辆路径问题

## 实验报告

可以直接前往 ![Overleaf](https://www.overleaf.com/read/rvhdcymrzbkv#c2b6b2) 阅读。

## 使用方法

### Step1. 安装环境

推荐使用conda管理环境，请事先安装某种conda。

```bash
git clone https://github.com/czyarl/ga_for_cvrptw.git # 克隆仓库
cd ga_for_cvrptw # 进入仓库目录
conda create -n ga_for_cvrptw python=3.12 # 创建环境。不过python版本影响应该不大
conda activate ga_for_cvrptw # 激活环境
pip install -r requirements.txt # 安装依赖
```

### Step2. 准备数据

这是数据集主页： ![vrptw](https://www.sintef.no/projectweb/top/vrptw/100-customers/) 。
或者可以直接点击 ![download](https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip) 下载。并请将之解压后，其中的 `In` 文件夹放到和 `main.py` 同级的目录下。

### Step2. 运行实验

可以直接依次运行以下两个文件以复现报告中的所有实验并获得相应折线图：

```bash
python run_all.py
python draw.py
```

如果想自己运行试验，也可以如下使用 `main.py` ：
```bash
python main.py \
	--problem cvrptw \
	--solver dp_ga \ 
	--data_folder ./In \
	--seed 42 \
	--population_size 200 \
	--generations 5000 \
	--mutation_rate 0.2 \
	--crossover_rate 0.9 \
	--disable_list "" \
	--use_data 10
``` 

参数说明：
- `problem`：问题类型，只可选 `cvrptw`
- `solver`：求解器类型，可选 `dp_ga` 或 `ga`，分别对应动态规划遗传算法与朴素遗传算法
- `data_folder`：数据文件夹路径，默认为 `./In`
- `seed`：随机种子，默认为 `42`
- `population_size`：种群大小，默认为 `100`
- `generations`：迭代次数，默认为 `1000`
- `mutation_rate`：变异概率，默认为 `0.2`
- `crossover_rate`：交叉概率，默认为 `0.9`
- `disable_list`：禁用某些限制（默认所有限制都考虑）。禁用的限制用英文逗号隔开，可选的包括demand(载重约束)、time(时间窗口约束)
- `use_data`：使用多少个数据，默认为 `10`

