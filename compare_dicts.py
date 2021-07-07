import ast


d = ast.literal_eval(open("conversion_dict.txt", "r").read())
d_new = ast.literal_eval(open("new_conv_dict.txt", "r").read())

for k in d_new.keys():
    if d[k] != d_new[k]:
        print("Difference at key: %s" % k)
        print("Old dict has value: %s" % d[k])
        print("New dict has value: %s" % d_new[k])