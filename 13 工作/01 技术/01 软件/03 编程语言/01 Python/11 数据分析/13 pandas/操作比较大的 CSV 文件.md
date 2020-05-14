
# 操作比较大的 CSV 文件

very large CSV files (6 gigabytes+)

6g 大的csv 文件。

如果想围绕这些数据长时间处理，那么还是需要加载进 数据库的，比如mySQL, postgreSQL

如果只是简单的分析，测试，那么可以用 Python, pandas and sqllite.

举例：

```py
import pandas as pd
from sqlalchemy import create_engine
file = '/path/to/csv/file'
print pd.read_csv(file, nrows=5)

csv_database = create_engine('sqlite:///csv_database.db')
chunksize = 100000
i = 0
j = 1
for df in pd.read_csv(file, chunksize=chunksize, iterator=True):
      df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})
      df.index += j
      i+=1
      df.to_sql('table', csv_database, if_exists='append')
      j = df.index[-1] + 1

df = pd.read_sql_query('SELECT * FROM table', csv_database)
df = pd.read_sql_query('SELECT COl1, COL2 FROM table where COL1 = SOMEVALUE', csv_database)
```

我们经常用 pandas 的 dataframe 来看一个小集合的数据，但是对于这个6g 的数据，我们可以用 sqllite 来看一个小集合的数据。

我们读一部分 csv 文件，然后写入 sqllite。

这时候，我们可以用 pandas 的 sql 把数据从 sqllite 拉出，不用担心内存问题。
