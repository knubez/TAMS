"""
create YMD strings that can be used to a certain subset of Kelly data


"""

import datetime as dt


deltat = dt.timedelta(days=1)

start = dt.datetime(year=2008, month=8, day=01)
end   = dt.datetime(year=2008, month=9, day=30)

outfname = '2008-August-September.txt'

current = start
with open(outfname, 'w') as f:
    while current <= end:
        f.write(current.strftime('%Y%m%d') + '\n')
        current += deltat
