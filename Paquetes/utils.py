import sys
import time

def print_loading_bar(iterations: int,message : str, efect_time : float = 0.5):
    for i in range(iterations):
        sys.stdout.write('\r')
        sys.stdout.write(message)
        sys.stdout.write(" [%-10s] %d%%" % ('=' * i, 10 * i))
        sys.stdout.flush()
        time.sleep(efect_time)




def test():
    print("Testing model:")
    print_loading_bar(iterations = 11,message= 'vbchevcb', efect_time = 0.5)
    print("\nTesting complete!")
    
    
if __name__ == "__main__":
    test()