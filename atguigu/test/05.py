# import time
# def loggerdecorator(func):
#     print("我爱你杨幂")
#     def wrapper(*args, **kwargs):
#         print(f"函数的入参信息是{args}")
#         result = func(*args, **kwargs)
#         print(f"函数的返回值是{result}")
#         return result
#     return wrapper
#
#
# def timerdecorator(func):
#     print("我爱你赵丽颖")
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"函数执行的耗时是{end_time - start_time}")
#         return result
#     return wrapper
#
# @loggerdecorator
# @timerdecorator
# def add(a,b):
#     return a+b
#
#
# print(add(10, 20))


from functools import wraps
def loggerdecorator(func):
    print("我爱你杨幂")
    @wraps(func)  #帮你把元数据恢复
    def wrapper(*args, **kwargs):
        print(f"函数的入参信息是{args}")
        result = func(*args, **kwargs)
        print(f"函数的返回值是{result}")
        return result
    return wrapper

@loggerdecorator
def add(a,b):
    return a+b


print(add(1, 2))