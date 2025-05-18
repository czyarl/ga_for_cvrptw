from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> dict:
        """解析数据文件并返回结构化数据"""
        pass

class CVRPTWDataset(Dataset):
    """带时间窗的容量约束车辆路径问题数据集"""
    
    def parse(self, file_path: str) -> dict:
        data = {
            'instance_name': '', # 实例名称
            'vehicles': {'number': 0, 'capacity': 0}, # 车辆信息
            'customers': [], # 所有客户信息，具体格式需要看下面的代码，懒得在这里说了
            'depot': None # 仓库信息，这里认为编号为0的就是仓库，所有车辆必须从仓库出发，返回仓库。
        }
        
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        data['instance_name'] = lines[0]
        lines = lines[1:]

        current_section = None
        for line in lines:
            if line == 'VEHICLE':
                current_section = 'VEHICLE'
                continue
            elif line == 'CUSTOMER':
                current_section = 'CUSTOMER'
                continue

            if current_section == 'VEHICLE':
                if 'NUMBER' in line:
                    continue  # 跳过标题行！！！
                parts = list(map(int, line.split()))
                data['vehicles']['number'] = parts[0]
                data['vehicles']['capacity'] = parts[1]
            
            elif current_section == 'CUSTOMER':
                if 'CUST NO.' in line:
                    continue  # 跳过标题行！！！
                
                parts = list(map(float, line.split()))
                if len(parts) != 7:
                    continue # 跳过标题行之后的那个空行
                
                customer = {
                    'id': int(parts[0]),
                    'x': parts[1],
                    'y': parts[2],
                    'demand': parts[3],
                    'ready_time': parts[4],
                    'due_time': parts[5],
                    'service_time': parts[6]
                }
                
                if customer['id'] == 0:
                    data['depot'] = customer
                else:
                    data['customers'].append(customer)
        
        return data