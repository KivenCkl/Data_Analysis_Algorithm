"""
NotQuiteABase
-------------
一个类似于数据库的 Python 实现，用于理解 SQL

对外提供九个方法：

1. insert(row_values)
插入数据

2. update(updates, predicate)
更新数据

3. delete(predicate=lambda row: True)
删除行

4. select(keep_columns=None, additional_columns=None)
数据库查询

5. where(predicate=lambda row: True)
返回仅满足特定条件的行

6. limit(num_rows)
仅返回前 num_rows 行

7. group_by(group_by_columns, aggregates, having=None)
将在特定列有相同值的行进行分组，并求出特定的汇总值

8. order_by(order)
按特定规则进行排序

9. join(other_table, left_join=False)
只对两表有共同列的部分做合并
"""
from collections import defaultdict


class Table:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def __repr__(self):
        return str(self.columns) + "\n" + "\n".join(map(str, self.rows))

    def insert(self, row_values):
        """插入数据

        Params
        ------
        row_values: 包含所有字段对应值的列表
        """
        if len(row_values) != len(self.columns):
            raise TypeError("wrong number of elements")
        row_dict = dict(zip(self.columns, row_values))
        self.rows.append(row_dict)

    def update(self, updates, predicate):
        """更新数据

        Params
        ------
        updates: dict，键为需要更新的列，值为这些字段的新值
        predicate: 对需要更新的列返回 True，否则返回 False
        """
        for row in self.rows:
            if predicate(row):
                for column, new_value in updates.items():
                    row[column] = new_value

    def delete(self, predicate=lambda row: True):
        """删除行

        Params
        ------
        predicate: 删除对应的行，默认删除所有行
        """
        self.rows = [row for row in self.rows if not (predicate(row))]

    def select(self, keep_columns=None, additional_columns=None):
        """数据库查询

        Params
        ------
        keep_columns: 声明希望在结果中保留的列名，默认包含所有列
        additional_columns: dict，键为新列名，值为计算新列值的函数，默认无操作

        Return
        ------
        New Table
        """
        if keep_columns is None:  # 没有没有指定列
            keep_columns = self.columns  # 则返回所有列

        if additional_columns is None:
            additional_columns = {}

        # 新表
        result_table = Table(keep_columns + list(additional_columns.keys()))

        for row in self.rows:
            new_row = [row[column] for column in keep_columns]
            for column_name, calculation in additional_columns.items():
                new_row.append(calculation(row))
            result_table.insert(new_row)

        return result_table

    def where(self, predicate=lambda row: True):
        """返回仅满足特定条件的行

        Params
        ------
        predicate: 条件函数

        Return
        ------
        New Table
        """
        where_table = Table(self.columns)
        where_table.rows = filter(predicate, self.rows)
        return where_table

    def limit(self, num_rows):
        """仅返回前 num_rows 行

        Params
        ------
        num_rows: int

        Return
        ------
        New Table
        """
        limit_table = Table(self.columns)
        limit_table.rows = self.rows[:num_rows]
        return limit_table

    def group_by(self, group_by_columns, aggregates, having=None):
        """将在特定列有相同值的行进行分组，并求出特定的汇总值

        Params
        ------
        group_by_columns: list，分组的列名
        aggregates: dict，对每组运行的汇总函数
        having: 作用于多行的可选判定函数

        Return
        ------
        New Table
        """
        grouped_rows = defaultdict(list)

        # 填充组
        for row in self.rows:
            key = tuple(row[column] for column in group_by_columns)
            grouped_rows[key].append(row)

        # 结果表中包含组列与汇总
        result_table = Table(group_by_columns + list(aggregates.keys()))

        for key, rows in grouped_rows.items():
            if having is None or having(rows):
                new_row = list(key)
                for _, aggregate_fn in aggregates.items():
                    new_row.append(aggregate_fn(rows))
                result_table.insert(new_row)

        return result_table

    def order_by(self, order):
        """按特定规则进行排序

        Params
        ------
        order: 条件函数

        Return
        ------
        New Table
        """
        new_table = self.select()  # 进行一次复制
        new_table.rows.sort(key=order)
        return new_table

    def join(self, other_table, left_join=False):
        """只对两表有共同列的部分做合并

        Params
        ------
        other_table: other Table
        left_join: bool，是否进行左并集

        Return
        ------
        New Table
        """
        # 两个表共同列
        join_on_columns = [c for c in self.columns if c in other_table.columns]

        # 右表中独有的列
        additional_columns = [
            c for c in other_table.columns if c not in join_on_columns
        ]

        # 左表中所有列 + 右表增加的列
        join_table = Table(self.columns + additional_columns)

        for row in self.rows:

            def is_join(other_table):
                return all(other_row[c] == row[c] for c in join_on_columns)

            other_rows = other_table.where(is_join).rows

            # 每对匹配的行生成一个新行
            for other_row in other_rows:
                join_table.insert([row[c] for c in self.columns] +
                                  [other_row[c] for c in additional_columns])

            # 如果没有行匹配，在左并集的操作下生成空值
            if left_join and not other_rows:
                join_table.insert([row[c] for c in self.columns] +
                                  [None for c in additional_columns])

        return join_table
