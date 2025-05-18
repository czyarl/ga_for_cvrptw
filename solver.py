from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
import random
import pygad  # 使用pygad库实现遗传算法
from tqdm import tqdm

class Solver(ABC):
    @abstractmethod
    def solve(self, data: Dict) -> Dict:
        """求解问题并返回解决方案"""
        pass

class GA_VRP_Solver(Solver):
    """遗传算法求解器。基因设计为 [a, b, c, 0, e, f, g, ..., 0, w, r, p] """
    
    def __init__(self, 
        population_size=100,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.9,
        disable_list=[],
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.disable_list = disable_list
        self.name = "GA_VRP_Solver"
        
    def solve(self, data: Dict) -> Dict:
        vehicle_capacity = data['vehicles']['capacity']
        vehicle_number = data['vehicles']['number']
        customers = data['customers']
        depot = data['depot']
        
        def fitness_func(cur_ga, solution, solution_idx):
            return self._calculate_fitness(solution, customers, depot, vehicle_capacity, vehicle_number)
        
        # 生成初始种群
        initial_population = self._generate_initial_population(customers, vehicle_capacity, vehicle_number)
        
        # 遗传算法配置
        ga_instance = pygad.GA(
            num_generations=self.generations, # 迭代的次数
            num_parents_mating=int(self.population_size/2), # 参与繁殖的父代数量
            fitness_func=fitness_func,
            
            sol_per_pop=self.population_size, # 每一代种群中的个体数量
            gene_type=int,
            initial_population=initial_population,
            
            mutation_probability=self.mutation_rate,
            crossover_probability=self.crossover_rate,
            crossover_type=self._custom_crossover,
            mutation_type=self._custom_mutation,
            
            suppress_warnings=True, 
            stop_criteria=["saturate_500"],
        )
        
        # 引入新的库开始计时，同时用一下进度条
        import time
        tot_time = 0
        best_fitness_history = []
        with tqdm(total=ga_instance.num_generations) as pbar:
            def on_generation(cur_ga: pygad.GA):
                pbar.update(1)
                best_fitness = cur_ga.best_solution()[1]
                pbar.set_postfix(best_fitness=best_fitness)
                best_fitness_history.append(float(best_fitness))

            ga_instance.on_generation = on_generation
            start = time.perf_counter()
            ga_instance.run()
            end = time.perf_counter()
            tot_time += end-start

        solution, solution_fitness, _ = ga_instance.best_solution()
        actual_rounds = ga_instance.generations_completed
        result = self._check_solution(solution, customers, depot, vehicle_capacity, vehicle_number, return_dict=True)
        result['time'] = tot_time
        result['population_size'] = self.population_size
        result['generations'] = self.generations
        result['actual_generations'] = actual_rounds
        result['mutation_rate'] = self.mutation_rate
        result['crossover_rate'] = self.crossover_rate
        result['best_fitness_history'] = best_fitness_history
        result['name'] = self.name
        result['disable_list'] = self.disable_list
        return result
    
    def _generate_initial_population(self, customers: List[Dict], capacity: int, number: int) -> List[List[int]]:
        """生成初始种群"""
        population = []
        num_customers = len(customers)
        
        for _ in range(self.population_size):
            routes = [[] for _ in range(number)]
            candidates = list(range(len(routes)))
            for cust_id in range(1, num_customers+1):
                curr_v = np.random.choice(candidates)
                routes[curr_v].append(cust_id)
                # NOTE: 暂时不检查车辆容量，保证初始解多样性。同时验证遗传算法调整能力
                # 以后可以加上，但是其实没太大必要。
                
            curr_population = []
            for idx, route in enumerate(routes):
                random.shuffle(route)
                if idx != 0:
                    route = [0] + route
                curr_population.extend(route)
            population.append(curr_population)
            
        return population

    def _check_solution(self, 
        solution: List[int], customers: List[Dict], 
        depot: Dict, capacity: int, number: int, 
        return_dict: bool = False
    ) -> Tuple[int, bool] | Dict:
        """计算路径总距离以及解的合法性。然后最后统计答案也用的是这个函数，所以加了个return_dict"""
        valid = True
        invalid_dist = 0
        
        routes = self._split_routes(solution, customers, depot, capacity, number)
        if routes is None:
            if return_dict:
                return {
                    'valid': False, 
                    'routes': [],
                    'distance': -9e7,
                }
            else:
                return -9e7, False
        if len(routes) > number:
            valid = False
            invalid_dist = -len(routes)
        
        total_distance = 0
        exceed_time = 0
        exceed_demand = 0
        visit_times = {cust_id: 0 for cust_id in range(1, len(customers)+1)}
        if return_dict:
            result_routes = []
        for route in routes:
            route_distance = 0
            curr_time = -20000
            prev = depot
            total_demand = 0
            route_valid = True
            
            for cust_id in route:
                visit_times[cust_id] += 1
                customer = customers[cust_id-1]  # 客户ID从1开始
                curr_distance = np.hypot(customer['x']-prev['x'], customer['y']-prev['y'])
                
                curr_time += curr_distance / 1 # speed is always 1
                if curr_time < customer['ready_time']:
                    curr_time = customer['ready_time']
                if curr_time > customer['due_time']:
                    if not 'time' in self.disable_list:
                        route_valid = False
                        exceed_time += curr_time - customer['due_time']
                curr_time += customer['service_time']
                
                route_distance += curr_distance
                total_demand += customer['demand']
                prev = customer
            
            # 检查路径容量、回归时间
            if total_demand > capacity:
                if not 'demand' in self.disable_list:
                    route_valid = False
                    exceed_demand += total_demand - capacity
            
            curr_distance = np.hypot(depot['x']-prev['x'], depot['y']-prev['y'])
            curr_time += curr_distance / 1
            if curr_time > depot['due_time']:
                if not 'time' in self.disable_list:
                    route_valid = False
                    exceed_time += curr_time - depot['due_time']
            route_distance += curr_distance
            
            # 合并路径信息
            total_distance += route_distance
            if not route_valid:
                valid = False
            if return_dict:
                result_routes.append({
                    'route': str(route), 
                    'distance': route_distance,
                    'valid': route_valid,
                })
                
        for cust_id, times in visit_times.items():
            if times != 1:
                valid = False
                invalid_dist = min(invalid_dist, -3e7)
        
        invalid_dist = min(invalid_dist, -exceed_demand-exceed_time)
        
        # print(f"valid = {valid}, total_distance = {total_distance}")
        # input()
        
        if return_dict:
            return {
                'valid': valid,
                'routes': result_routes,
                'distance': total_distance if valid else invalid_dist,
            }
        return total_distance if valid else invalid_dist, valid

    def _calculate_fitness(self, solution: List[int], customers: List[Dict], depot: Dict, capacity: int, number: int) -> float:
        """计算解的适应度: (路径总距离+1)的倒数"""
        dist, valid = self._check_solution(solution, customers, depot, capacity, number, return_dict=False)
        if not valid:
            return dist/1000 if dist <= 0 else -1e8
        return 10000 / (dist+1)
    
    def _split_routes(self, solution: List[int], customers: List[Dict], depot: Dict, capacity: int, number: int) -> List[List[int]]:
        """将基因序列分割为有效路径，不进行过多的检查"""
        routes = []
        current_route = []
        for gene in solution:
            if gene == 0:
                routes.append(current_route.copy())
                current_route = []
            else:
                current_route.append(int(gene))
        routes.append(current_route)
        return routes
    
    def _custom_crossover(self, parents, offspring_size, ga_instance):
        """交叉操作"""
        # print(f"in crossover, len parents = {len(parents)}, offspring_size = {offspring_size}")
        # input()
        offspring = []
        def extract_pos_and_order(route: List[int]) -> Tuple[List[int], List[int]]:
            pos = [0 if i == 0 else 1 for i in route]
            order = [i for i in route if i != 0]
            return pos, order
        def merge_pos_and_order(pos: List[int], order: List[int]) -> List[int]:
            res = []
            curr_idx = 0
            for i in pos:
                if i == 0:
                    res.append(0)
                else:
                    res.append(order[curr_idx])
                    curr_idx += 1
            return res
        for _ in range(offspring_size[0]):
            # 检查是否跳过交叉
            if np.random.rand() > ga_instance.crossover_probability:
                # 直接随机选择一个父代作为子代
                parent = random.choice(parents)
                offspring.append(parent.copy())
                continue
            
            parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
            
            pos1, order1 = extract_pos_and_order(parent1)
            pos2, order2 = extract_pos_and_order(parent2)
            
            point1 = np.random.randint(0, len(order1))
            point2 = np.random.randint(point1, len(order1)+1)
            child_order = order1[point1:point2].copy()
            p2_remain = [i for i in order2 if i not in child_order]
            child_order = p2_remain[:point1] + child_order + p2_remain[point1:]
            
            child_pos = random.choice([pos1, pos2])
            
            child = merge_pos_and_order(child_pos, child_order)
            offspring.append(child)
            
        return np.array(offspring)
    
    def _custom_mutation(self, offspring, ga_instance):
        """变异。为了保持解的合法性，这里选择交换两个基因的位置。输入 ndarray, size 为 (batch_size, gene_length)"""
        for idx in range(offspring.shape[0]):
            if np.random.rand() < self.mutation_rate:
                i, j = np.random.choice(offspring.shape[1], 2, replace=False)
                offspring[idx, [i, j]] = offspring[idx, [j, i]]
        return offspring