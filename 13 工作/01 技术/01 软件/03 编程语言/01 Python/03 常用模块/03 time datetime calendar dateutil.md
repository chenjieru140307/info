
# 可以补充进来的

相关：

- time
- datetime
- calendar
- dateutil


## time

- [文档](https://docs.python.org/3/library/time.html#module-time)


举例：


```py
import time

print(time.time())
print(time.localtime(time.time()))
print(time.asctime(time.localtime(time.time())))
print()

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
print()

a = "Sat Mar 28 22:24:24 2016"
print(time.mktime(time.strptime(a,"%a %b %d %H:%M:%S %Y")))
```

输出：

```txt
1589467757.2960143
time.struct_time(tm_year=2020, tm_mon=5, tm_mday=14, tm_hour=22, tm_min=49, tm_sec=17, tm_wday=3, tm_yday=135, tm_isdst=0)
Thu May 14 22:49:17 2020

2020-05-14 22:49:17
Thu May 14 22:49:17 2020

1459175064.0
```


说明：

- `time.time()` 得到当前时间戳。时间戳单位最适合做日期运算。但是 1970 年之前的日期就无法以此表示了。2038 年以后的日期也无法表示。（没有很清楚）
- `time.localtime(time.time())` 将时间戳转为 `struct_time` 元组：
  - tm_year  2008
  - tm_mon   1 到 12
  - tm_mday  1 到 31
  - tm_hour  0 到 23
  - tm_min   0 到 59
  - tm_sec   0 到 61 (60或 61 是闰秒)
  - tm_wday  0到 6 (0是周一)
  - tm_yday  1 到 366(儒略历)
  - tm_isdst -1, 0, 1, -1是决定是否为夏令时的旗帜 
- `asctime()` 可以简单的获取可读的时间。
- `time.strftime` 方法来格式化日期

- `time.mktime(time.strptime(a,"%a %b %d %H:%M:%S %Y")` 将格式字符串转换为时间戳

格式化符号：

- %y 两位数的年份表示（00-99）
- %Y 四位数的年份表示（000-9999）
- %m 月份（01-12）
- %d 月内中的一天（0-31）
- %H 24小时制小时数（0-23）
- %I 12小时制小时数（01-12）
- %M 分钟数（00=59）
- %S 秒（00-59）
- %a 本地简化星期名称
- %A 本地完整星期名称
- %b 本地简化的月份名称
- %B 本地完整的月份名称
- %c 本地相应的日期表示和时间表示
- %j 年内的一天（001-366）
- %p 本地 A.M。或 P.M。的等价符
- %U 一年中的星期数（00-53）星期天为星期的开始
- %w 星期（0-6），星期天为星期的开始
- %W 一年中的星期数（00-53）星期一为星期的开始
- %x 本地相应的日期表示
- %X 本地相应的时间表示
- %Z 当前时区的名称
- %% %号本身


time 函数：

- [time.altzone](https://www.w3cschool.cn/Python/att-time-altzone.html) 返回格林威治西部的夏令时地区的偏移秒数。如果该地区在格林威治东部会返回负值（如西欧，包括英国）。对夏令时启用地区才能使用。
- [time.asctime([tupletime])](https://www.w3cschool.cn/Python/att-time-asctime.html) 接受时间元组并返回一个可读的形式为"Tue Dec 11 18:07:14 2008"（2008年 12 月 11 日 周二 18 时 07 分 14 秒）的 24 个字符的字符串。
- [time.clock( )](https://www.w3cschool.cn/Python/att-time-clock.html) 用以浮点数计算的秒数返回当前的 CPU 时间。用来衡量不同程序的耗时，比 time.time()更有用。
- [time.ctime([secs])](https://www.w3cschool.cn/Python/att-time-ctime.html) 作用相当于 asctime(localtime(secs))，未给参数相当于 asctime()
- [time.gmtime([secs])](https://www.w3cschool.cn/Python/att-time-gmtime.html) 接收时间辍（1970纪元后经过的浮点秒数）并返回格林威治天文时间下的时间元组 t。注：t.tm_isdst始终为 0.
- [time.localtime([secs])](https://www.w3cschool.cn/Python/att-time-localtime.html) 接收时间辍（1970纪元后经过的浮点秒数）并返回当地时间下的时间元组 t（t.tm_isdst可取 0 或 1，取决于当地当时是不是夏令时）。
- [time.mktime(tupletime)](https://www.w3cschool.cn/Python/att-time-mktime.html) 接受时间元组并返回时间辍（1970纪元后经过的浮点秒数）。
- [time.sleep(secs)](https://www.w3cschool.cn/Python/att-time-sleep.html) 推迟调用线程的运行，secs指秒数。
- [time.strftime(fmt[,tupletime])](https://www.w3cschool.cn/Python/att-time-strftime.html) 接收以时间元组，并返回以可读字符串表示的当地时间，格式由 fmt 决定。
- [time.strptime(str,fmt='%a %b %d %H:%M:%S %Y')](https://www.w3cschool.cn/Python/att-time-strptime.html) 根据 fmt 的格式把一个时间字符串解析为时间元组。
- [time.time( )](https://www.w3cschool.cn/Python/att-time-time.html) 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。
- [time.tzset()](https://www.w3cschool.cn/Python/att-time-tzset.html) 根据环境变量 TZ 重新初始化时间相关设置。

time 属性：

- **time.timezone** 属性 time.timezone是当地时区（未启动夏令时）距离格林威治的偏移秒数（>0，美洲;<=0大部分欧洲，亚洲，非洲）。
- **time.tzname** 属性 time.tzname包含一对根据情况的不同而不同的字符串，分别是带夏令时的本地时区名称，和不带的。


## Calendar




- 用来处理年历和月历
- [文档](https://docs.python.org/3/library/calendar.html#module-calendar)

举例：


```py
import calendar

cal = calendar.month(2016, 1)
print(cal)
```


以上实例输出结果：

```
    January 2016
Mo Tu We Th Fr Sa Su
             1  2  3
 4  5  6  7  8  9 10
11 12 13 14 15 16 17
18 19 20 21 22 23 24
25 26 27 28 29 30 31
```




calendar 函数：


- **calendar.calendar(year,w=2,l=1,c=6)** 返回一个多行字符串格式的 year 年年历，3个月一行，间隔距离为 c。 每日宽度间隔为 w 字符。每行长度为 21* W+18+2* C。l是每星期行数。
- **calendar.firstweekday( )** 返回当前每周起始日期的设置。默认情况下，首次载入 caendar 模块时返回 0，即星期一。
- **calendar.isleap(year)** 是闰年返回 True，否则为 false。
- **calendar.leapdays(y1,y2)** 返回在 Y1，Y2两年之间的闰年总数。
- **calendar.month(year,month,w=2,l=1)** 返回一个多行字符串格式的 year 年 month 月日历，两行标题，一周一行。每日宽度间隔为 w 字符。每行的长度为 7* w+6。l是每星期的行数。
- **calendar.monthcalendar(year,month)** 返回一个整数的单层嵌套列表。每个子列表装载代表一个星期的整数。Year年 month 月外的日期都设为 0；范围内的日子都由该月第几日表示，从 1 开始。
- **calendar.monthrange(year,month)** 返回两个整数。第一个是该月的星期几的日期码，第二个是该月的日期码。日从 0（星期一）到 6（星期日）；月从 1 到 12。
- **calendar.prcal(year,w=2,l=1,c=6)** 相当于 print calendar.calendar(year,w,l,c).
- **calendar.prmonth(year,month,w=2,l=1)** 相当于 print calendar.calendar（year，w，l，c）。
- **calendar.setfirstweekday(weekday)** 设置每周的起始日期码。0（星期一）到 6（星期日）。 
- **calendar.timegm(tupletime)** 和 time.gmtime相反：接受一个时间元组形式，返回该时刻的时间辍（1970纪元后经过的浮点秒数）。
- **calendar.weekday(year,month,day)** 返回给定日期的日期码。0（星期一）到 6（星期日）。月份为 1（一月） 到 12（12月）。

## datetime

- [文档](https://docs.Python.org/library/datetime.html#module-datetime)


举例：


```py
from datetime import datetime

dt = datetime(2011, 10, 29, 20, 30, 21)
print(dt.day)
print(dt.minute)
print(dt.date())
print(dt.time())
print(dt.strftime('%m/%d/%Y %H:%M'))
dt.replace(minute=0, second=0)
print(dt)
print()

print(datetime.strptime('20091031', '%Y%m%d'))
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
print()

dt1 = datetime(2011, 10, 29, 20, 30, 21)
dt2 = datetime(2011, 11, 15, 22, 30)
delta = dt2 - dt1
print(delta)
print(type(delta))
print(dt1)
print(dt1 + delta)
```

输出：

```
29
30
2011-10-29
20:30:21
10/29/2011 20:30
2011-10-29 20:30:21

2009-10-31 00:00:00
2020-05-14 23:07:48
2020-05-14 23:07:48.859925

17 days, 1:59:39
<class 'datetime.timedelta'>
2011-10-29 20:30:21
2011-11-15 22:30:00
```

说明:

- datetime.datetime是不可变的，所以 `dt.replace(minute=0, second=0)` 是新创建了一个 object。
- 两个不同的 datetime object能产生一个 datetime.timedelta类型：
- `time.time()` 来作为时间戳只能精确到秒。`datetime.datetime.now()` 可以精确到毫秒。

## dateutil

- [文档](https://labix.org/Python-dateutil)



