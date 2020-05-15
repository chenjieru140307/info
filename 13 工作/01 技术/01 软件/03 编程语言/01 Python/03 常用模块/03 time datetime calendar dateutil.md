
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
from datetime import datetime, timedelta, timezone

dt = datetime.now()  # 获取当前datetime
print(dt)
print(type(dt))
print(dt)
print()

dt = datetime(2011, 10, 29, 20, 30, 21)
print(dt.day)
print(dt.minute)
print(dt.date())
print(dt.time())
dt.replace(minute=0, second=0)
print(dt)
print()

dt = datetime(2011, 10, 29, 20, 30, 21)
print(dt.timestamp())  # 把datetime转换为timestamp
t = 1429417200.0
print(datetime.fromtimestamp(t))  # 时间戳转为本地时间
print(datetime.fromtimestamp(t).tzinfo)
print(datetime.utcfromtimestamp(t))  # 时间戳转为UTC时间
print(datetime.utcfromtimestamp(t).tzinfo)
print()

dt = datetime.now()
print(datetime.strptime('20091031', '%Y%m%d'))
print(datetime.strptime('20091031', '%Y%m%d').tzinfo)
print(datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S'))
print(dt.strftime('%a, %b %d %H:%M'))
print(dt.strftime('%Y-%m-%d %H:%M:%S'))
print(dt.strftime('%Y-%m-%d %H:%M:%S.%f'))
print(dt.strftime('%m/%d/%Y %H:%M'))
print()

dt1 = datetime(2011, 10, 29, 20, 30, 21)
dt2 = datetime(2011, 11, 15, 22, 30)
delta = dt2 - dt1
print(delta)
print(type(delta))
print(dt1)
print(dt1 + delta)
now = datetime.now()
print(now)
print(now + timedelta(hours=10))
print(now - timedelta(days=1))
print(now + timedelta(days=2, hours=12))
print()


tz_utc_8 = timezone(timedelta(hours=8))  # 创建时区UTC+8:00
now = datetime.now()
print(now)
dt = now.replace(tzinfo=tz_utc_8)  # 强制设置为UTC+8:00
print(dt)
print()


# 拿到UTC时间，并强制设置时区为UTC+0:00:
utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
print(utc_dt)
# astimezone()将转换时区为北京时间:
bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
print(bj_dt)
# astimezone()将转换时区为东京时间:
tokyo_dt = utc_dt.astimezone(timezone(timedelta(hours=9)))
print(tokyo_dt)
# astimezone()将bj_dt转换时区为东京时间:
tokyo_dt2 = bj_dt.astimezone(timezone(timedelta(hours=9)))
print(tokyo_dt2)
```

输出：

```
2020-05-15 10:57:53.732067
<class 'datetime.datetime'>
2020-05-15 10:57:53.732067

29
30
2011-10-29
20:30:21
2011-10-29 20:30:21

1319891421.0
2015-04-19 12:20:00
None
2015-04-19 04:20:00
None

2009-10-31 00:00:00
None
2015-06-01 18:19:59
Fri, May 15 10:57
2020-05-15 10:57:53
2020-05-15 10:57:53.732067
05/15/2020 10:57

17 days, 1:59:39
<class 'datetime.timedelta'>
2011-10-29 20:30:21
2011-11-15 22:30:00
2020-05-15 10:57:53.739033
2020-05-15 20:57:53.739033
2020-05-14 10:57:53.739033
2020-05-17 22:57:53.739033

2020-05-15 10:57:53.739033
2020-05-15 10:57:53.739033+08:00

2020-05-15 02:57:53.739033+00:00
2020-05-15 10:57:53.739033+08:00
2020-05-15 11:57:53.739033+09:00
2020-05-15 11:57:53.739033+09:00
```

说明:


- 在计算机中，时间实际上是用数字表示的。我们把1970年1月1日 00:00:00 UTC+00:00时区的时刻称为 epoch time，记为0（1970年以前的时间timestamp为负数），当前时间就是相对于epoch time的秒数，称为timestamp。
- timestamp 的值与时区毫无关系，因为 timestamp 一旦确定，其 UTC 时间就确定了，转换到任意时区的时间也是完全确定的，这就是为什么计算机存储的当前时间是以 timestamp 表示的，因为全球各地的计算机在任意时刻的 timestamp 都是完全相同的（假定时间已校准）。
- 注意到timestamp是一个浮点数，它没有时区的概念，而在转化为 datetime 时，是考虑时区的。
  - `datetime.fromtimestamp(t)` 是在 timestamp 和本地时间做转换。本地时间是指当前操作系统设定的时区。例如北京时区是东8区，则本地时间：`2015-04-19 12:20:00` 就是UTC+8:00 时区的时间：`2015-04-19 12:20:00 UTC+8:00` ，而此刻的格林威治标准时间与北京时间差了8小时，也就是UTC+0:00时区的时间应该是：`2015-04-19 04:20:00 UTC+0:00`
  - `datetime.utcfromtimestamp(t)` 可以直接将时间戳转为 UTC+0:00 标准时区时间 `2015-04-19 04:20:00 UTC+0:00`。
  - 转换后的 datetime 的时区属性 `tzinfo` 仍是 None。（为什么呢？为什么没有把信息附加进来。）
- 使用 `strptime` 转换后的 datetime 的时区属性 `tzinfo` 也是 None。
- `now.replace(tzinfo=tz_utc_8)` 可以强制设置时区属性，最好不要这样设定。
- 可以先通过 `utcnow()` 拿到当前的UTC时间，然后用 `astimezone` 转换为任意时区的时间。
- `datetime.datetime` 是不可变的，所以 `dt.replace(minute=0, second=0)` 是新创建了一个 object。
- `time.time()` 来作为时间戳只能精确到秒。`datetime.datetime.now()` 可以精确到毫秒。
- 注意：`datetime` 表示的时间需要时区信息才能确定一个特定的时间，否则只能视为本地时间。如果要存储 datetime，最佳方法是将其转换为 `timestamp` 再存储，因为 timestamp 的值与时区完全无关。

## dateutil

- [文档](https://labix.org/Python-dateutil)



