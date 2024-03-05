import sqlite3
import array
from io import StringIO

class DB:
    def __init__(self, db_name):
        self._db_name = db_name
        self._conn = sqlite3.connect(':memory:')
        with open(db_name, 'r', encoding='ISO-8859-1') as f:
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
    
    def get_clauses_by_tokens(self, tokens, target_value, accumlation):
        if target_value == 1:
            operator = '> 0'
        else:
            operator = '< 0'

        # Create a comma-separated string of tokens for use in the SQL query
        token_string = ','.join(str(token) for token in tokens)

        self._cursor.execute(
            '''SELECT weight, literals FROM clauses WHERE token IN ({}) AND weight {} ORDER BY RANDOM() LIMIT ?'''.format(token_string, operator),
            (accumlation,)
        )
        clauses = self._cursor.fetchall()
        return clauses

    
    def close_connection(self):
        self._conn.close()
