import functools
import torch
import torch.nn.functional as F
import types
import time
import pprint
import contextlib
from collections import defaultdict


# 创建一个字典来存储调用次数
call_count = {}
# 用于跟踪被装饰的函数和装饰次数
decorated_count = defaultdict(int)
# 备份原始函数
original_functions = {}

# 要跳过装饰的属性列表
SKIP_ATTRIBUTES = [
    'storage', 'untyped_storage', 'typed_storage',  # 跳过可能触发 TypedStorage 警告的属性
    'reduce_op',  # 跳过 reduce_op (已弃用)
]

def count_calls(func, module_name=None):
    @functools.wraps(func)
    def wrapper_count_calls(*args, **kwargs):
        full_name = module_name + '.' + func.__name__
        
        # 记录开始时间
        start_time = time.time()
        result = func(*args, **kwargs)
        # 记录结束时间
        end_time = time.time()
        
        # 计算调用时间
        elapsed_time = end_time - start_time
        
        if full_name not in call_count:
            call_count[full_name] = {"count": 0, "total_time": 0.0}
        
        call_count[full_name]["count"] += 1
        call_count[full_name]["total_time"] += elapsed_time
        
        return result
    
    wrapper_count_calls._is_decorated = True
    wrapper_count_calls._original_func = func
    return wrapper_count_calls

def decorate_function(module, attr_name):
    """装饰单个函数"""
    # 跳过已知会引起警告的属性
    if attr_name in SKIP_ATTRIBUTES:
        return False
    
    if hasattr(module, attr_name):
        attr = getattr(module, attr_name)
        key = f"{module.__name__}.{attr_name}"
        
        if callable(attr):
            # 如果函数已经被装饰，只增加引用计数
            if hasattr(attr, '_is_decorated'):
                decorated_count[key] += 1
                return True
            
            # 否则装饰函数并保存原始函数
            if key not in original_functions:
                original_functions[key] = attr
            
            decorated_attr = count_calls(original_functions[key], module.__name__)
            setattr(module, attr_name, decorated_attr)
            decorated_count[key] = 1
            return True
    
    return False

def restore_function(module, attr_name):
    """恢复单个原始函数"""
    # 跳过已知会引起警告的属性
    if attr_name in SKIP_ATTRIBUTES:
        return False
    
    key = f"{module.__name__}.{attr_name}"
    
    # 减少引用计数
    if key in decorated_count:
        decorated_count[key] -= 1
        
        # 只有当引用计数为0时才恢复原始函数
        if decorated_count[key] <= 0:
            if key in original_functions:
                setattr(module, attr_name, original_functions[key])
                del decorated_count[key]
            return True
    
    return False

def should_skip_module(module):
    """检查是否应该跳过特定模块"""
    module_name = getattr(module, '__name__', '')
    # 跳过 torch.distributed 模块，因为它包含已弃用的 reduce_op
    if module_name.startswith('torch.distributed'):
        return True
    return False

def decorate_module(module, recursive=True, visited=None):
    """装饰模块中的所有函数"""
    if visited is None:
        visited = set()
    
    module_id = id(module)
    if module_id in visited:
        return
    visited.add(module_id)
    
    # 跳过可能引起警告的模块
    if should_skip_module(module):
        return
    
    for attr_name in dir(module):
        # 跳过已知会引起警告的属性
        if attr_name in SKIP_ATTRIBUTES:
            continue
            
        try:
            attr = getattr(module, attr_name)
            
            # 装饰函数
            if isinstance(attr, types.FunctionType):
                decorate_function(module, attr_name)
            
            # 装饰可调用对象
            elif callable(attr) and not isinstance(attr, type):
                decorate_function(module, attr_name)
            
            # 递归装饰子模块
            elif recursive and isinstance(attr, types.ModuleType):
                module_name = getattr(attr, '__name__', '')
                if module_name.startswith('torch') and not should_skip_module(attr):
                    decorate_module(attr, recursive=True, visited=visited)
            
            # 装饰类
            elif isinstance(attr, type):
                decorate_class(attr)
                
        except AttributeError:
            continue

def decorate_class(cls):
    """装饰类中的方法"""
    for attr_name in dir(cls):
        # 跳过已知会引起警告的属性
        if attr_name in SKIP_ATTRIBUTES:
            continue
            
        try:
            attr = getattr(cls, attr_name)
            if isinstance(attr, types.FunctionType):
                decorate_function(cls, attr_name)
            elif attr_name in ['__add__', '__mul__', '__sub__', '__truediv__', '__matmul__', '__pow__', '__mod__']:
                # 特殊处理运算符重载方法
                decorate_function(cls, attr_name)
        except (AttributeError, TypeError):
            continue

def restore_module(module, recursive=True, visited=None):
    """恢复模块中的所有原始函数"""
    if visited is None:
        visited = set()
    
    module_id = id(module)
    if module_id in visited:
        return
    visited.add(module_id)
    
    # 跳过可能引起警告的模块
    if should_skip_module(module):
        return
        
    for attr_name in dir(module):
        # 跳过已知会引起警告的属性
        if attr_name in SKIP_ATTRIBUTES:
            continue
            
        try:
            # 尝试恢复函数
            restore_function(module, attr_name)
            
            # 递归恢复子模块
            attr = getattr(module, attr_name)
            if recursive and isinstance(attr, types.ModuleType):
                module_name = getattr(attr, '__name__', '')
                if module_name.startswith('torch') and not should_skip_module(attr):
                    restore_module(attr, recursive=True, visited=visited)
            elif isinstance(attr, type):
                restore_class(attr)
                
        except AttributeError:
            continue

def restore_class(cls):
    """恢复类中的原始方法"""
    for attr_name in dir(cls):
        # 跳过已知会引起警告的属性
        if attr_name in SKIP_ATTRIBUTES:
            continue
            
        try:
            restore_function(cls, attr_name)
        except (AttributeError, TypeError):
            continue

# 重置计数器
def reset_counters():
    """重置所有计数器"""
    global call_count
    call_count.clear()

# 获取统计结果
def get_statistics():
    """获取当前的统计结果"""
    return call_count

# 打印统计结果
def print_statistics(sort_by="count", top_n=None):
    """打印当前的统计结果
    
    Args:
        sort_by: 排序依据，可以是"count"或"total_time"
        top_n: 只显示前N个结果，默认显示所有
    """
    if not call_count:
        print("No statistics available.")
        return
        
    # 对结果进行排序
    sorted_stats = sorted(call_count.items(), 
                          key=lambda x: x[1][sort_by], 
                          reverse=True)
    
    if top_n is not None:
        sorted_stats = sorted_stats[:top_n]
    
    print(f"{'Function Name':<50} {'Count':<10} {'Total Time (s)':<15} {'Avg Time (s)':<15}")
    print("-" * 90)
    for name, stats in sorted_stats:
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        print(f"{name:<50} {stats['count']:<10} {stats['total_time']:<15.6f} {avg_time:<15.6f}")

# 上下文管理器，用于临时启用监控
class Monitor:
    def __init__(self, modules=None, reset=False):
        """
        Args:
            modules: 要监控的模块列表，默认为[torch]
            reset: 是否在进入上下文前重置计数器
        """
        self.modules = modules if modules is not None else [torch]
        self.reset = reset
    
    def __enter__(self):
        if self.reset:
            reset_counters()
        
        # 装饰指定模块中的函数
        for module in self.modules:
            decorate_module(module)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始函数
        for module in self.modules:
            restore_module(module)

# 创建一个更简单的上下文管理器函数
@contextlib.contextmanager
def monitor(modules=None, reset=False):
    """上下文管理器，用于在特定代码块中启用监控
    
    Args:
        modules: 要监控的模块列表，默认为[torch]
        reset: 是否在进入上下文前重置计数器
    """
    with Monitor(modules=modules, reset=reset):
        yield


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # 创建一个简单的神经网络
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleNN()
    x = torch.randn(32, 10)

    # 这部分代码不会被监控
    y = model(x)

    # 使用上下文管理器监控特定代码块
    with monitor(reset=True):
        y = model(x)

    # 打印统计结果
    print_statistics(sort_by="total_time")

    # 继续累积统计信息（不重置计数器）
    with monitor():
        for _ in range(10):
            y = model(x)

    # 打印累积的统计结果
    print_statistics(sort_by="count", top_n=5)

    # 只监控特定模块
    with monitor(modules=[torch.nn.functional], reset=True):
        y = model(x)

    print_statistics()