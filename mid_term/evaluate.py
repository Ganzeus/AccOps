import functools
import torch
import torch.nn.functional as F
import types
import time
import pprint
# 创建一个字典来存储调用次数
call_count = {}

def count_calls(func, module_name=None):
    @functools.wraps(func)
    def wrapper_count_calls(*args, **kwargs):
        # module_name = func.__module__ if hasattr(func, '__module__') and func.__module__ else 'torch'
        full_name = module_name + '.' + func.__name__
        # call_count[full_name] = call_count.get(full_name, 0) + 1
        # print(f"Function {full_name} called {call_count[full_name]} times")
        # return func(*args, **kwargs)
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
    return wrapper_count_calls

def set_new_attr(module, attr_name, attr):
    if not hasattr(attr, "_is_decorated"):
        decorated_attr = count_calls(attr, module.__name__)
        decorated_attr._is_decorated = True
        setattr(module, attr_name, decorated_attr)

# 递归封装所有的包
def auto_decorate_module(module, visited=None):
    if visited is None:
        visited = set()
    
    module_name = module.__name__
    if module_name in visited:
        return
    visited.add(module_name)
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
            # if isinstance(attr, types.FunctionType):
            if isinstance(attr, types.FunctionType):
                set_new_attr(module, attr_name, attr)
                # print(f"Decorated function: {module_name}.{attr_name}")
            elif isinstance(attr, types.ModuleType) and attr.__name__.startswith('torch'):
                # print(f"Descending into module: {attr.__name__}")
                auto_decorate_module(attr, visited)
            elif isinstance(attr, type):
                # print(f"Descending into class: {attr.__name__} in {module_name}")
                auto_decorate_class(attr)
            elif callable(attr):
                set_new_attr(module, attr_name, attr)
        except AttributeError:
            continue


def auto_decorate_class(cls):
    for attr_name in dir(cls):
        # if attr_name.startswith('__') and attr_name.endswith('__'):
        #     continue  # Skip special attributes
        try:
            attr = getattr(cls, attr_name)
            if isinstance(attr, types.FunctionType):
                set_new_attr(cls, attr_name, attr)
            elif attr_name in ['__add__', '__mul__', '__sub__', '__truediv__', '__matmul__', '__pow__', '__mod__']:
                # 特殊处理运算符重载方法
                set_new_attr(cls, attr_name, attr)
        except (AttributeError, TypeError) as e:
            continue

auto_decorate_module(torch)


##### 填入测试代码 #######
# 创建一些张量
x = torch.randn(2, 2)
y = torch.randn(2, 2)


a = torch.add(x, y)
a = x + y
b = torch.mul(x, y)
b = x * y

result1 = F.relu(x)
result2 = F.softmax(x, dim=1)
result3 = F.relu(y)

# 再次调用 randn
# z = torch.randn(3, 3)

###### 获取结果 #######
pprint.pprint(call_count)
