import os
import json
import matplotlib.pyplot as plt
import numpy as np

def get_data(file_name, constraints, tar_features):
    with open(file_name, 'r') as f:
        # load jsonl file
        data = [json.loads(line) for line in f]
    filtered = {}
    for item in data:
        valid = True
        for key, value in constraints.items():
            if value is not None:
                if item[key] != value:
                    # print(f"{key} is not {value}: {item[key]}")
                    # input()
                    valid = False
        if valid:
            item_name = "_".join([str(item[key]) for key in tar_features])
            filtered[item_name] = item
    filtered = list(filtered.values())
    print(len(filtered))
    return filtered

def draw(file_name, tar_features, save_folder, constraints):
    import copy
    constraints = copy.deepcopy(constraints)
    real_name = {
        "name": "solver", 
        "population_size": 'population',
        "crossover_rate": 'crossover',
        "mutation_rate": 'mutation',
        "disable_list": 'w/o',
    }
    for tar_feature in tar_features:
        constraints[tar_feature] = None
    data = get_data(file_name, constraints, tar_features)
    data = sorted(data, key=lambda x: [x[t] for t in tar_features])
    # 把所有data中的best_fitness_history按照solver作为标签绘制到一张图折线图上。
    plt.figure()  # 确保使用新的图形
    plt.title(f"Comparison of {' '.join([real_name[i] for i in tar_features])}")
    
    # 绘制每条曲线并自动生成标签
    for item in data:
        history = item.get("best_fitness_history", [])
        if history:  # 确保数据有效
            label = ""
            for tar_feature in tar_features:
                if label != "": label += ";"
                label += f"{real_name[tar_feature]}:{item.get(tar_feature, 'N/A')}"
            linestyle = '-' if item['name'] == 'DP_GA_VRP_Solver' else '--'
            plt.plot(history, label=label, linestyle=linestyle)
    
    plt.legend()  # 自动收集标签，无需手动传递
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.tight_layout()  # 调整布局避免截断
    
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"compare_{"_".join(tar_features)}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

file_name = "./In/result/c203.jsonl"
save_folder = "./results"

constraints = {
    "name": "DP_GA_VRP_Solver", 
    "population_size": 200,
    "crossover_rate": 0.9,
    "mutation_rate": 0.2,
    "disable_list": ["time"],
}
draw(file_name, ["name", "population_size"], save_folder, constraints)
draw(file_name, ["name", "mutation_rate"], save_folder, constraints)
draw(file_name, ["name", "crossover_rate"], save_folder, constraints)

constraints = {
    "name": "DP_GA_VRP_Solver", 
    "population_size": 200,
    "crossover_rate": 0.9,
    "mutation_rate": 0.2,
    "disable_list": ["none"],
}
draw(file_name, ["population_size"], save_folder, constraints)
draw(file_name, ["mutation_rate"], save_folder, constraints)