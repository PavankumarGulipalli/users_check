def add_numbers(a,b):
    return a+b

def sub_numbers(a,b):
    return a-b

print(add_numbers(2,3))
print(sub_numbers(5,4))

def multiply_numbers(a,b):
    return a*b

print(multiply_numbers(2,3))

def divide_numbers(a,b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a/b

print(divide_numbers(10,2))
# Test division by zero
try:
    print(divide_numbers(5,0))
except ValueError as e:
    print(e)