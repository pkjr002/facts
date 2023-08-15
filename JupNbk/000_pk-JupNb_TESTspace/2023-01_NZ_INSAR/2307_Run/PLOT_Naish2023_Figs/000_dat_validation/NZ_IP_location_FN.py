    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def readTXT_head(path,filename):
    #
    with open(f'{path}/{filename}', 'r') as file:
        for i, line in enumerate(file):
            if i < 10:
                print(line.strip())
    return None

# .............................
def latlon_head(path,filename,lonIDX,latIDX):
    #
    import pandas as pd
    from io import StringIO
    #
    with open(f'{path}/{filename}', 'r') as f: lns = f.readlines()
    col_nme = lns[0].strip().split(', ')
    data    = "\n".join(lne.strip() for lne in lns[1:])
    df      = pd.read_csv(StringIO(data), delim_whitespace=True, names=col_nme, dtype=str)
    array   = df.to_numpy()
    #
    Lon = array[:,lonIDX].astype(float)
    Lat = array[:,latIDX].astype(float)
    return Lon,Lat

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def readTXT_csv(path,filename):
    #
    import csv
    with open(f'{path}/{filename}', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, row in enumerate(reader):
            if i < 10:
                print(" ".join(row))
    return None

# .............................
def latlon_csv(path, filename):
    #
    import pandas as pd
    file_path = f'{path}/{filename}'
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, dtype=str)
    array = df.to_numpy()
    #
    Lon = array[:, 1].astype(float)
    Lat = array[:, 2].astype(float)
    return Lon, Lat


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot(ax,var,linPARAM,label):
    #   
    import matplotlib.pyplot as plt
    #
    # ylab = axPARAM['ylab',None]
    # y_min = axPARAM['y_min',None]
    # y_max = axPARAM['y_max',None]
    # y_ticks = axPARAM['y_ticks',None]
    #
    # xlab = axPARAM['xlab',None]
    # x_min = axPARAM['x_min',None]
    # x_max = axPARAM['x_max',None]
    # x_ticks = axPARAM['x_ticks',None]
    #
    clr = linPARAM.get('clr', None)
    lab = linPARAM.get('lab', None)
    mrkr = linPARAM.get('mrkr', None)
    mrkrsz = linPARAM.get('mrkrsz', None)
    #
    title = label.get('title',None)
    #
    ax.plot(var,mrkr,markersize=mrkrsz,color=clr,label=lab)
    #
    plt.title(title,fontsize=28)
    lgd=plt.legend(loc='lower left',fontsize=18)
    for handle in lgd.legendHandles:
        handle.set_markersize(30) 