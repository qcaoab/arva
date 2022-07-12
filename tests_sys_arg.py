import sys

print("tracing parameter entered from terminal: ", sys.argv[1])

list_k = [float(item) for item in sys.argv[1].split(" ")]

[print(list_k)]