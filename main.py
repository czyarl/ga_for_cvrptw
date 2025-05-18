from dataset import CVRPTWDataset, Dataset
from solver import GA_VRP_Solver, Solver
from solver_2 import DP_GA_VRP_Solver

import argparse
import json
import os
from typing import List, Dict, Tuple
import random

def solve_file(file_name: str, Dataset: Dataset, Solver: Solver) -> Dict:
    data = Dataset.parse(file_name)
    result = Solver.solve(data)
    
    folder = os.path.dirname(file_name)
    file_name = os.path.basename(file_name).split('.')[0]
    
    result_folder = os.path.join(folder, "result")
    fig_folder = os.path.join(result_folder, file_name+"_fig")
    fig_name = f"mut{result['mutation_rate']}-cross{result['crossover_rate']}-pop{result['population_size']}-disable{"_".join(result['disable_list'])}-{result['name']}"
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(fig_folder, exist_ok=True)
    
    with open(os.path.join(result_folder, f"{file_name}.jsonl"), "a") as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")
        
    # 绘制result['best_fitness_history']的折线图
    import matplotlib.pyplot as plt
    plt.plot(result['best_fitness_history'])
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Best Fitness History of {file_name} by {fig_name}')
    plt.savefig(os.path.join(fig_folder, f"{fig_name}.png"))
    plt.close()
    return result

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="cvrptw", choices=['cvrptw'])
    parser.add_argument("--solver", type=str, default="ga", choices=['ga', 'dp_ga'])
    parser.add_argument("--data_folder", type=str, default="./In")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--mutation_rate", type=float, default=0.2)
    parser.add_argument("--crossover_rate", type=float, default=0.9)
    parser.add_argument("--disable_list", type=str, default="")
    args = parser.parse_args()
    
    set_seed(args.seed)
    if args.problem == "cvrptw":
        Dataset = CVRPTWDataset()
    else:
        raise ValueError(f"Unknown problem: {args.problem}")
    
    if args.solver == "ga":
        Solver = GA_VRP_Solver(
            population_size=args.population_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            disable_list=args.disable_list.split(","),
        )
    elif args.solver == "dp_ga":
        Solver = DP_GA_VRP_Solver(
            population_size=args.population_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            disable_list=args.disable_list.split(","),
        )
    else:
        raise ValueError(f"Unknown solver: {args.solver}")
    
    file_list = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder) if f.endswith(".txt")]
    file_list = sorted(file_list)
    random.shuffle(file_list)
    file_list = file_list[:10]
    tot_cnt = 0
    all_len = []
    all_time = []
    all_generations = []
    for file_name in file_list:
        result = solve_file(file_name, Dataset, Solver)
        tot_cnt += 1
        if result['valid']:
            all_len.append(result['distance'])
        all_time.append(result['time'])
        all_generations.append(result['actual_generations'])
    print(f"找到合法解的数量: {len(all_len)} ({len(all_len) / tot_cnt * 100:.2f}%)")
    print(f"找到合法解的平均最优解距离: {sum(all_len) / (len(all_len)+1e-8):.2f}")
    with open("result.jsonl", "a") as f:
        json.dump({
            "problem": args.problem,
            "solver": args.solver,
            "seed": args.seed,
            "population_size": args.population_size,
            "generations": args.generations,
            "mutation_rate": args.mutation_rate,
            "crossover_rate": args.crossover_rate,
            "disable_list": args.disable_list,
            "valid_cnt": len(all_len),
            "valid_ratio": len(all_len) / tot_cnt * 100,
            "avg_distance": sum(all_len) / (len(all_len)+1e-8),
            "avg_time": sum(all_time) / (len(all_time)+1e-8),
            "avg_generations": sum(all_generations) / (len(all_generations)+1e-8),
        }, f, ensure_ascii=False)
        f.write("\n")

if __name__ == "__main__":
    main()