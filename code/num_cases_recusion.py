import pickle as pkl

# def recusion(lower, upper, tot_type_0, depth, ar, file_path='count.pkl'):
#     # print(depth, tot_type_0)
#     if depth == tot_type_0 + 1:
#         curr_count = pkl.load(open(file_path, 'rb'))
#         curr_count += 1
#         pkl.dump(curr_count, open(file_path, 'wb'))
#     else:
#         for ind in range(lower, upper + 1):
#             # print(lower, upper)
#             recusion(ind, min(ar, upper + 1), tot_type_0, depth + 1, ar)

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

    file_path = 'count.pkl'
    pkl.dump(0, open(file_path, 'wb'))
    b = v + 1 - c
    upper = min(b, Ar)
    tot_type_0 = c - 1
    for ind_st in range(1, upper + 1):
        # print(ind_st)
        recusion(ind_st, min(Ar, upper + 1), tot_type_0, 2, Ar)

    curr_count = pkl.load(open(file_path, 'rb'))
    print(curr_count)

def give_back_num_options(init, UB, tot_type_0, Ar):

    file_path = 'count.pkl'
    pkl.dump(0, open(file_path, 'wb'))
    # b = v + 1 - c
    # upper = min(b, Ar)
    # tot_type_0 = c - 1
    if tot_type_0 == 0:
        return 1
    else:
        for ind_st in range(1, min(init, UB)+1):
            # print(ind_st)
            recusion(ind_st, min(init+1, UB), tot_type_0, 2, UB)

        curr_count = pkl.load(open(file_path, 'rb'))
        # print(curr_count)
        return curr_count


def main():
    total_options = 0

    v = 9
    c = 6
    Ar = 8
    b = v+1-c
    init = b
    UB = Ar
    tot_type_0 = c-1

    for Id_p in range(b, Ar-1+1):
        before_options = give_back_num_options(b, Id_p, Id_p-b, Id_p)
        after_options = give_back_num_options(1, Ar-Id_p, c-1-(Id_p-b)-1, Ar-Id_p)
        print(Id_p, before_options,after_options)
        total_options += before_options*after_options

    print(total_options)
    # give_back_num_options(init, UB, tot_type_0, Ar)
    # give_back_num_options_id_0(9, 9, 8)

if __name__ == '__main__':

    main()