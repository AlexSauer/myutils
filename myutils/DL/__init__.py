# If a module depend on some dependencies, only import if available
try:
    from myutils.DL.experiments import *
except ModuleNotFoundError as e:
    print(e)

try: 
    from myutils.DL.plotting import *
except ModuleNotFoundError as e:
    print(e)