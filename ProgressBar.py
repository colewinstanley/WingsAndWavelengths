# ProgressBar.py ; self-explanitory 
import time
import sys

def startProgress(title):
    global progress_x
    global st
    st = time.time()
    sys.stdout.write(title + ": [" + "-"*45 + "]" + chr(8)*46)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    x = int(x * 45 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x

def endProgress():
    end = time.time()
    sys.stdout.write("#" * (45 - progress_x) + "] time: %s sec\n" % str(end - st)[:5])
    sys.stdout.flush()