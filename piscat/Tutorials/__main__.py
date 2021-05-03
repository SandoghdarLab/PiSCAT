from .startup import start_tutorials

try:
    start_tutorials()
except Exception as e:
    print(e)
