fname = ".doit"

N = 10
NP = 5

f = open(fname, "w")

for num in range(N):
    f.write("echo ' '\n")
    f.write("echo ' '\n")
    f.write("echo 'BEG test " + str(num) + "  '\n")
    f.write("./.pypipe_workers " + str(NP) + " test_3d_ho.py\n")
    for k in range(1, NP):
        f.write("mv out_p" + str(k) + " out_" + str(num) + "_p" + str(k) + "\n")
    f.write("echo 'END test " + str(num) + "  '\n")
    f.write("echo ' '\n")
    f.write("echo ' '\n")
