date = require 'date'
t_end = '20:14:23.647996'
t_start = '20:08:12.576270'
local t2 = date(t_end)
local t1 = date(t_start)
local epoch = 10000
local seconds = date.diff(t2, t1):spanseconds()/epoch
print (seconds)
