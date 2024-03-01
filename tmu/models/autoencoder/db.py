import sqlite3
import array
from io import StringIO

class DB:
    def __init__(self, db_name):
        self._db_name = db_name
        self._conn = sqlite3.connect(':memory:')
        with open(db_name, 'r') as f:
            buffer = StringIO(f.read())
        self._conn.executescript(buffer.getvalue())
        # self._conn = sqlite3.connect(db_name)
        self._cursor = self._conn.cursor()

    def get_clauses_by_token(self, token, target_value, accumlation):
        if target_value == 1:
            operator = '> 0'
        else:
            operator = '< 0'

        self._cursor.execute(
            '''SELECT weight, literals FROM clauses WHERE token = ? AND weight ''' + operator + ''' ORDER BY RANDOM() LIMIT ?''',
            (int(token),accumlation)
        )
        clauses = self._cursor.fetchall()
        return clauses
    
    def close_connection(self):
        self._conn.close()
