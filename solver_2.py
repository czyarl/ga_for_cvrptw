from typing import List, Dict, Tuple
import numpy as np
import random

from solver import GA_VRP_Solver

from numba import njit, jit

point_type = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('demand', np.float64),
    ('ready_time', np.float64),
    ('due_time', np.float64),
    ('service_time', np.float64)
])

@njit
def _split_routes(solution: np.ndarray, customers: np.ndarray, capacity: int, number: int, disable_time: bool, disable_demand: bool) -> Tuple[np.ndarray, np.ndarray, int]:
    # 以下尽量使用0-n来代表个体，0是depot, i是solution[i-1]
    def get_info(x: int) -> Dict:
        return customers[solution[x-1]]
    def cal_dist(x: int, y: int):
        c1 = get_info(x)
        c2 = get_info(y)
        return np.hypot(c1["x"] - c2["x"], c1["y"] - c2["y"])
    
    length = solution.shape[0]
    
    demand_pre_sum = np.zeros(length+1, dtype=np.int32)
    dist_pre_sum = np.zeros(length+1, dtype=np.float32)
    dist_to_depot = np.zeros(length+1, dtype=np.float32)
    for i in range(1, length+1):
        demand_pre_sum[i] = demand_pre_sum[i-1] + get_info(i)['demand']
        dist_pre_sum[i] = dist_pre_sum[i-1] + cal_dist(i-1, i)
        dist_to_depot[i] = cal_dist(0, i)
    
    # 计算从每个客户开始依次遍历，最远能遍历到哪里
    far = np.full(length+1, length, dtype=np.int32)
    if not disable_time:
        for i in range(1, length+1):
            curr_time = -20000
            last_pt = 0
            for j in range(i, length+1):
                curr_time += cal_dist(last_pt, j) / 1
                if curr_time < get_info(j)['ready_time']:
                    curr_time = get_info(j)['ready_time']
                if curr_time > get_info(j)['due_time']:
                    far[i] = min(far[i], j-1)
                    break
                curr_time += get_info(j)['service_time']
                if curr_time + cal_dist(j, 0)/1 > get_info(0)['due_time']:
                    far[i] = min(far[i], j-1)
                    break
                last_pt = j
    if not disable_demand:
        cur_far_pos = 0
        for i in range(1, length+1):
            cur_far_pos = max(cur_far_pos, i-1)
            while cur_far_pos < length and demand_pre_sum[cur_far_pos+1]-demand_pre_sum[i-1] <= capacity:
                cur_far_pos += 1
            far[i] = min(far[i], cur_far_pos)
    # print(far)
    # input()
    for i in range(1, length+1):
        assert far[i] >= i-1
    # print(far)
    
    # 真的开始DP了，并记录最优转移来源，以方便提供路径划分方案
    ans = np.full((length+1, length+1), 1e8, dtype=np.float32)
    ans_src = np.full((length+1, length+1), 0, dtype=np.int32)
    queue = np.zeros(length+1, dtype=np.int32)
    ans[0][0] = 0
    for v in range(1, number+1):
        ans[v][0] = 0
        ans_src[v][0] = 0
        l = 0
        r = -1
        for i in range(1, length+1):
            # 维护单调队列
            while l <= r and ans[v-1][i-1]+dist_to_depot[i] <= ans[v-1][queue[r]]+dist_to_depot[queue[r]+1]:
                r -= 1
            r += 1
            queue[r] = i-1
            while l <= r and i > far[queue[l]+1]:
                l += 1
            
            # 更新答案
            ans[v][i] = ans[v-1][queue[l]] + dist_to_depot[queue[l]+1] + dist_to_depot[i] + dist_pre_sum[i] - dist_pre_sum[queue[l]+1]
            ans_src[v][i] = queue[l]

    if ans[number][length] <= 1e7:
        return ans, ans_src, number
    
    # ans2 = np.full((length+1, length+1), 1e8, dtype=np.float32)
    # ans_src2 = np.full((length+1, length+1), 0, dtype=np.int32)
    # ans2[:number+1, :] = ans
    # ans_src2[:number+1, :] = ans_src 
    for v in range(number+1, length+1):
        ans[v][0] = 0
        ans_src[v][0] = 0
        l = 0
        r = -1
        for i in range(1, length+1):
            # 维护单调队列
            while l <= r and ans[v-1][i-1]+dist_to_depot[i] <= ans[v-1][queue[r]]+dist_to_depot[queue[r]+1]:
                r -= 1
            r += 1
            queue[r] = i-1
            while l <= r and i > far[queue[l]+1]:
                l += 1
            
            # 更新答案
            ans[v][i] = ans[v-1][queue[l]] + dist_to_depot[queue[l]+1] + dist_to_depot[i] + dist_pre_sum[i] - dist_pre_sum[queue[l]+1]
            ans_src[v][i] = queue[l]
        if ans[v][length] <= 1e7:
            return ans, ans_src, v
    return ans, ans_src, number

class DP_GA_VRP_Solver(GA_VRP_Solver):
    """使用了动态规划的遗传算法求解器。基因设计为 [a, b, c, e, f, g, ..., w, r, p] ，分割位置使用DP求解"""
    def __init__(self, **args):
        super().__init__(**args)
        self.name = "DP_GA_VRP_Solver"
    
    def _generate_initial_population(self, customers: List[Dict], capacity: int, number: int) -> List[List[int]]:
        """生成初始种群。和原版的区别就是去掉了其中的0"""
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
                curr_population.extend(route)
            population.append(curr_population)
            
        return population
    
    def _split_routes(self, solution: List[int], customers: List[Dict], depot: Dict, capacity: int, number: int) -> List[List[int]]:
        """将基因序列分割为有效路径。这里使用DP寻找最优分割位置"""
        solution_array = np.array(solution, dtype=np.int32)
        customers_array = np.array([
            (c['x'], c['y'], c['demand'], c['ready_time'], c['due_time'], c['service_time'])
            for c in [depot] + customers
        ], dtype=point_type)
        disable_time = "time" in self.disable_list
        disable_demand = "demand" in self.disable_list
        
        ans, ans_src, vehicle_used = _split_routes(solution_array, customers_array, capacity, number, disable_time, disable_demand)
        
        if ans[vehicle_used][len(solution)] > 1e7:
            return None # 当前访问序列无解
        
        routes = []
        current_route = []
        curr_customer = len(solution)
        for v in range(vehicle_used, 0, -1):
            # print(f"v = {v}, curr_customer = {curr_customer}")
            prev_customer = ans_src[v][curr_customer]
            if prev_customer < curr_customer:
                current_route = [solution[i-1] for i in range(prev_customer+1, curr_customer+1)]
                routes.append(current_route)
            else:
                routes.append([])
            curr_customer = prev_customer
        # print(routes)
        # input()
        return routes
    
    def _custom_crossover(self, parents, offspring_size, ga_instance):
        """交叉操作。应该是叫OX交叉。"""
        # print(f"in crossover, len parents = {len(parents)}, offspring_size = {offspring_size}")
        # input()
        offspring = []
        for _ in range(offspring_size[0]):
            # 检查是否跳过交叉
            if np.random.rand() > ga_instance.crossover_probability:
                # 直接随机选择一个父代作为子代
                parent = random.choice(parents)
                offspring.append(parent.copy().tolist())
                continue
            
            parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
            parent1 = parent1.tolist()
            parent2 = parent2.tolist()
            
            point1 = np.random.randint(0, len(parent1))
            point2 = np.random.randint(point1, len(parent1)+1)
            child = parent1[point1:point2].copy()
            p2_remain = [i for i in parent2 if i not in child]
            child = p2_remain[:point1] + child + p2_remain[point1:]

            offspring.append(child)
            
        return np.array(offspring)