import pickle as pkl
import pandas as pd
from utils_ph import *
from tqdm import tqdm


def recusion(lower, upper, tot_type_0, depth, UB, file_path='count.pkl'):
    # print(depth, tot_type_0)
    if depth == tot_type_0 + 1:
        curr_count = pkl.load(open(file_path, 'rb'))
        curr_count += 1
        pkl.dump(curr_count, open(file_path, 'wb'))
    else:
        for ind in range(lower, upper + 1):
            # print(lower, upper)
            recusion(ind, min(UB, upper + 1), tot_type_0, depth + 1, UB)

def give_back_num_options_id_0(v,c,Ar):

    if c == v+1:
        return 0
    elif c==0 or c==1:
        return 1
    else:
        file_path = 'count.pkl'
        pkl.dump(0, open(file_path, 'wb'))
        b = v + 1 - c
        upper = min(b, Ar)
        tot_type_0 = c - 1
        for ind_st in range(1, upper + 1):
            # print(ind_st)
            recusion(ind_st, min(Ar, upper + 1), tot_type_0, 2, Ar)

        curr_count = pkl.load(open(file_path, 'rb'))
        return curr_count

def give_back_num_options(init, UB, tot_type_0, Ar):

    file_path = 'count.pkl'
    pkl.dump(0, open(file_path, 'wb'))

    if tot_type_0 <= 0:
        return 1
    else:
        for ind_st in range(1, min(init, UB)+1):
            # print(ind_st)
            recusion(ind_st, min(init+1, UB), tot_type_0, 2, UB)

        curr_count = pkl.load(open(file_path, 'rb'))
        # print(curr_count)
        return curr_count

def one_idle_period(c,b,Ar):

    '''
    :param c: number of future arrivals
    :param b: number of existing customers
    :param Ar: arrival epoch of the second type 1
    :return: total_options: the total count of combintations
    '''
    if c == 0:
        return 1
    elif c == 1:
        return 1
    elif Ar == c+b:
        Id_p = b+c-1
        before_options = give_back_num_options(b, Id_p, Id_p - b, Id_p)
        return before_options # there is only ome after option
    else:
        total_options = 0
        for Id_p in range(b, Ar):
            before_options = give_back_num_options(b, Id_p, Id_p-b, Id_p)
            after_options = give_back_num_options(1, Ar-Id_p, c-1-(Id_p-b)-1, Ar-Id_p)
            # print(Id_p, before_options, after_options)
            total_options += before_options*after_options
        return total_options

def add_row_to_df(df,v,c,Id,Ar, number):
    curr_row = df.shape[0]
    df.loc[curr_row, 'v'] = v
    df.loc[curr_row, 'c'] = c
    df.loc[curr_row, 'Id'] = Id
    df.loc[curr_row, 'Ar'] = Ar
    df.loc[curr_row, 'number'] = number
    return df

def main():


    ub_v = 11


    df = pd.DataFrame([], columns=['v', 'c', 'Id', 'Ar', 'number'])
    df = add_row_to_df(df, 0, 0, 0, 0, 1)
    df = add_row_to_df(df, 0, 1, 1, 1, 1)


    for v in tqdm(range(1,ub_v)):
        for c in range(v+2):
            if c == 0:
                Ar = 0
                total_id_0 = give_back_num_options_id_0(v, c, Ar)
                df = add_row_to_df(df,v ,c ,0 , Ar, total_id_0)
            elif c == v+1:
                pass
            else:
                for Ar in range(1, v+1):
                    # print(v, c, Ar)
                    total_id_0 = give_back_num_options_id_0(v, c, Ar)
                    # print(total_id_0)
                    df = add_row_to_df(df, v, c, 0, Ar, total_id_0)


    for v in tqdm(range(1, ub_v)):
        for c in range(v + 2):
            b = v + 1 - c
            if c == 0:
                pass
            elif c == v + 1:
                for Ar in range(b + 1, v+1):
                    # print(v, c, Ar)
                    total_id_1 = give_back_num_options_id_0(v, v, Ar)
                    # print(total_id_1)
                    df = add_row_to_df(df, v, c, 1, Ar, total_id_1)
            elif c == 1:
                Ar = v+1
                # print(v, c, Ar)
                total_id_1 = one_idle_period(c, b, Ar)
                # print(total_id_1)
                df = add_row_to_df(df, v, c, 1, Ar, total_id_1)
            else:
                for Ar in range(b+1, v + 2):
                    # print(v, c, Ar)
                    total_id_1 = one_idle_period(c, b, Ar)
                    # print(total_id_1)
                    df = add_row_to_df(df, v, c, 1, Ar, total_id_1)

    unique_v = df['v'].unique()
    for v in unique_v:
        unique_c = df.loc[df['v'] == v, 'c'].unique()
        for c in unique_c:
            if c > 0:
                max_Id = c
                for Id in range(1, max_Id):
                    ar_set = df.loc[(df['v'] == v) & (df['c'] == c) & (df['Id'] == Id), 'Ar'].unique()
                    for ind, ar in enumerate(ar_set):
                        if ind < len(ar_set) - 1:
                            df_row = df.shape[0]
                            df.loc[df_row, 'v'] = v
                            df.loc[df_row, 'c'] = c
                            df.loc[df_row, 'Id'] = Id + 1
                            df.loc[df_row, 'Ar'] = ar + 1
                            df.loc[df_row, 'number'] = df.loc[
                                (df['v'] == v) & (df['c'] == c) & (df['Id'] == Id) & (
                                            df['Ar'] == ar), 'number'].values[0]
                        else:
                            if (c == v + 1) & (Id == 1):
                                df_row = df.shape[0]
                                df.loc[df_row, 'v'] = v
                                df.loc[df_row, 'c'] = c
                                df.loc[df_row, 'Id'] = Id + 1
                                df.loc[df_row, 'Ar'] = ar + 1
                                df.loc[df_row, 'number'] = df.loc[
                                    (df['v'] == v) & (df['c'] == c) & (df['Id'] == Id) & (
                                                df['Ar'] == ar), 'number'].values[0]

    pkl.dump(df, open('df', 'wb'))


if __name__ == '__main__':

    main()