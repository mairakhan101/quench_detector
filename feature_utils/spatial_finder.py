theta = 9.5/ 180 * np.pi
width = 8
k = np.tan(theta)
dy = width / np.cos(theta)
size = (dy * 10 / np.tan(theta), dy * 10)

def locate(io, lr):
    def sub_locate(i, j):    # i: slope > 0, j: slope < 0
        '''
        y = k * x + dy * (11 - i)
        y = -k * x + dy * (j - 11) + size[1]
        '''
        x = (dy * (i + j - 22) + size[1]) / (2 * k)
        y = k * x + dy * (11 - i)
        return x, y
        
    io_parts = io.split('_')
    lr_parts = lr.split('_')
    
    x = 0
    y = 0
    if io_parts[0] == 'IN' and lr_parts[0] == 'LE':
        i = int(io_parts[2][1:])
        l = int(lr_parts[1][1:])
        x, y = sub_locate(l, i)
    elif io_parts[0] == 'OUT' and lr_parts[0] == 'LE':
        o = int(io_parts[1][1:])
        l = int(lr_parts[1][1:])
        x, y = sub_locate(l, o)
    elif io_parts[0] == 'IN' and lr_parts[0] == 'RE':
        i = int(io_parts[1][1:])
        r = int(lr_parts[1][1:])
        x, y = sub_locate(i, r)
        x += size[0]
    elif io_parts[0] == 'OUT' and lr_parts[0] == 'RE':
        o = int(io_parts[2][1:])
        r = int(lr_parts[1][1:])
        x, y = sub_locate(o, r)
        x += size[0]
    return x, y

def rhombus(x, y):
    return [x - dy / 2 / k, x, x + dy / 2 / k, x, x - dy / 2 / k], [y, y + dy / 2, y, y - dy / 2, y]
