# coding=utf-8

def s(a):
    if a>=0.5:
        return 1
    elif abs(a)<0.5:
        return 0
    else:
        return -1

def circle(a):
    tick = 1
    while tick<10:
        b = round(a*255.0)
        print "{}: {:b} {}".format(tick+1, int(b), b)
        a = 2*a-s(2*a)
        tick += 1

def main():
    # a = -0.270588235294
    a = 162
    print("{:b} {}".format(a, a))
    a = a/255.0
    a = a - s(a) # 一阶残差
    circle(a)

if __name__ == '__main__':
    main()
