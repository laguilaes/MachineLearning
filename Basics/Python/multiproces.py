from functools import partial
import multiprocessing as mp
import pandas as pd
import numpy as np
import time

def fun(input_df, input_df2, col):
    return input_df[col].mean() * input_df2[col].mean()

def foo_pool(x):
    time.sleep(2)
    return x*x

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)


if __name__ == "__main__":

    #De forma no secuencial
    df = pd.DataFrame({"a": np.linspace(0, 10, 9),
                        "b": np.logspace(1, 5, 9)})
    cols = df.columns.to_list()
    fun_p = partial(fun, df, df)
    pool = mp.Pool(3)
    res = pool.map(fun_p, cols)
    pool.close()
    pool.join()
    print(res)


    #De forma secuencial
    pool = mp.Pool()
    for i in range(10):
        pool.apply_async(foo_pool, args = (i, ), callback = log_result)
    pool.close()
    pool.join()
    print(result_list)
