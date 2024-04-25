import time
def execution_time(func : callable = None):
    """
    Funcion decoradora que calcula el tiempo de ejecucion de la funcion que esta decorando
    Parameters
    ----------
        - func : (callable) funcion a decorar
    return
    ------
        - wrapper : (callable) funcion "envoltorio2 que agrega la funcionalidad del calculo de tiempo de ejecucion
    """
    def wrapper(*args, **kwargs):
        start_execution = time.time()
        return_func = func(*args,**kwargs)
        end_execution = time.time()
        print(f'Exexcution time of {func.__name__} {end_execution - start_execution}')
        return return_func
    return wrapper

"""
# EJEMPLO DE USO
# funcion de prueba decorada
@execution_time
def sleep():
    time.sleep(2)


if __name__ == '__main__':
    sleep()

"""

