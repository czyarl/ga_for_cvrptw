import os
from itertools import product

all_solver = ['dp_ga', 'ga']
all_population_size = [200, 30, 100, 400]
all_generations = [5000]
all_mutation_rate = [0.2, 0, 0.5]
all_crossover_rate = [0.9, 0, 0.4]
# all_disable_list = ['time', 'none']

for mutation_rate in all_mutation_rate:
    os.system(f'python main.py \
        --solver {all_solver[1]} \
        --population_size {all_population_size[0]} \
        --generations {all_generations[0]} \
        --mutation_rate {mutation_rate} \
        --crossover_rate {all_crossover_rate[0]} \
        --disable_list none')

for mutation_rate in all_mutation_rate:
    os.system(f'python main.py \
        --solver {all_solver[0]} \
        --population_size {all_population_size[0]} \
        --generations {all_generations[0]} \
        --mutation_rate {mutation_rate} \
        --crossover_rate {all_crossover_rate[0]} \
        --disable_list none')
for population_size in all_population_size:
    os.system(f'python main.py \
        --solver {all_solver[0]} \
        --population_size {population_size} \
        --generations {all_generations[0]} \
        --mutation_rate {all_mutation_rate[0]} \
        --crossover_rate {all_crossover_rate[0]} \
        --disable_list none')

for solver in all_solver:
    os.system(f'python main.py \
        --solver {solver} \
        --population_size {all_population_size[0]} \
        --generations {all_generations[0]} \
        --mutation_rate {all_mutation_rate[0]} \
        --crossover_rate {all_crossover_rate[0]} \
        --disable_list time')

    for population_size in all_population_size[1:]:
        os.system(f'python main.py \
            --solver {solver} \
            --population_size {population_size} \
            --generations {all_generations[0]} \
            --mutation_rate {all_mutation_rate[0]} \
            --crossover_rate {all_crossover_rate[0]} \
            --disable_list time')

    for mutation_rate in all_mutation_rate[1:]:
        os.system(f'python main.py \
            --solver {solver} \
            --population_size {all_population_size[0]} \
            --generations {all_generations[0]} \
            --mutation_rate {mutation_rate} \
            --crossover_rate {all_crossover_rate[0]} \
            --disable_list time')
        
    for crossover_rate in all_crossover_rate[1:]:
        os.system(f'python main.py \
            --solver {solver} \
            --population_size {all_population_size[0]} \
            --generations {all_generations[0]} \
            --mutation_rate {all_mutation_rate[0]} \
            --crossover_rate {crossover_rate} \
            --disable_list time')